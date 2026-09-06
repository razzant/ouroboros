# tests/conftest.py — shared pytest fixtures for the Ouroboros test suite.
#
# Loaded automatically by pytest before any test module runs.
# Cross-module helpers that are not pytest fixtures (e.g. SDK mock, extension
# runtime cleanup) live in ``tests/_shared.py`` instead.
import asyncio
import functools
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import pytest
pytest.register_assert_rewrite("tests.ui_media_delivery_smoke")


_PYTEST_DATA_DIR = None
# Repo root for a live-DATA run, which has no pytest data dir to hang it off. Created lazily
# so the hermetic lane never leaves an unused temp dir behind (see pytest_sessionfinish).
_PYTEST_REPO_FALLBACK = None
if os.environ.get("OUROBOROS_ALLOW_LIVE_DATA_TESTS") != "1":
    _LIVE_DATA_ROOT = (
        os.environ.get("OUROBOROS_TEST_LIVE_DATA_ROOT")
        or os.environ.get("OUROBOROS_DATA_DIR")
        or str(pathlib.Path.home() / "Ouroboros" / "data")
    )
    _PYTEST_DATA_DIR = pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-pytest-data-"))
    os.environ["OUROBOROS_PYTEST_ACTIVE"] = "1"
    os.environ["OUROBOROS_TEST_LIVE_DATA_ROOT"] = _LIVE_DATA_ROOT
    os.environ["OUROBOROS_DATA_DIR"] = str(_PYTEST_DATA_DIR)
    os.environ["OUROBOROS_SETTINGS_PATH"] = str(_PYTEST_DATA_DIR / "settings.json")
    # Conftest-WIDE bench-runs isolation. devtools benchmark tests invoke
    # run_*.main(), whose run_root() defaults to the real <repo>/../bench_runs
    # when OUROBOROS_BENCH_RUNS_ROOT is unset — leaking timestamped run dirs and
    # ouroboros_task_body.json stubs into the operator's bench_runs/ (the
    # programbench/swe_bench_pro pollution). A file-local autouse fixture only
    # covered one module; pinning it here covers every test.
    os.environ["OUROBOROS_BENCH_RUNS_ROOT"] = str(_PYTEST_DATA_DIR / "bench_runs")


_ORIGINAL_POPEN_INIT = subprocess.Popen.__init__
_PYTEST_CHILD_DATA_DIR = os.environ.get("OUROBOROS_DATA_DIR", "")
_PYTEST_CHILD_LIVE_ROOT = os.environ.get("OUROBOROS_TEST_LIVE_DATA_ROOT", "")
_PYTEST_CHILD_BENCH_ROOT = os.environ.get("OUROBOROS_BENCH_RUNS_ROOT", "")
_PYTEST_POPEN_PATCHED = False


def _isolated_child_env(value) -> dict:
    child_env = dict(value)
    if not child_env.get("OUROBOROS_DATA_DIR"):
        child_env["OUROBOROS_DATA_DIR"] = _PYTEST_CHILD_DATA_DIR
    if not child_env.get("OUROBOROS_SETTINGS_PATH"):
        child_env["OUROBOROS_SETTINGS_PATH"] = str(
            pathlib.Path(child_env["OUROBOROS_DATA_DIR"]) / "settings.json"
        )
    if _PYTEST_CHILD_BENCH_ROOT and not child_env.get("OUROBOROS_BENCH_RUNS_ROOT"):
        child_env["OUROBOROS_BENCH_RUNS_ROOT"] = _PYTEST_CHILD_BENCH_ROOT
    child_env["OUROBOROS_PYTEST_ACTIVE"] = "1"
    child_env["OUROBOROS_TEST_LIVE_DATA_ROOT"] = _PYTEST_CHILD_LIVE_ROOT
    return child_env


def _install_pytest_child_isolation() -> None:
    """Keep the disposable data root when a test scrubs a child env."""
    global _PYTEST_POPEN_PATCHED
    if _PYTEST_DATA_DIR is None or _PYTEST_POPEN_PATCHED:
        return

    @functools.wraps(_ORIGINAL_POPEN_INIT)
    def isolated_init(self, *args, **kwargs):
        positional = list(args)
        if len(positional) > 10 and positional[10] is not None:
            positional[10] = _isolated_child_env(positional[10])
        elif kwargs.get("env") is not None:
            kwargs["env"] = _isolated_child_env(kwargs["env"])
        return _ORIGINAL_POPEN_INIT(self, *positional, **kwargs)

    subprocess.Popen.__init__ = isolated_init
    _PYTEST_POPEN_PATCHED = True


def _restore_pytest_child_isolation() -> None:
    global _PYTEST_POPEN_PATCHED
    if _PYTEST_POPEN_PATCHED:
        subprocess.Popen.__init__ = _ORIGINAL_POPEN_INIT
        _PYTEST_POPEN_PATCHED = False


def _bind_pytest_repo_root() -> None:
    """Point git_ops.REPO_DIR away from the operator's live checkout.

    Unbound, git_ops.REPO_DIR (no env fallback) sends
    update_merge._update_tx_marker_path() at the LIVE repo's .git, so a staged managed merge
    blocks the whole suite through the registry guard. An empty dir with no .git makes the
    strict read `absent` — the honest allow. Direct assignment: init() would also rewrite
    BRANCH_DEV/BRANCH_STABLE.

    Keyed on the REPO opt-in (OUROBOROS_ALLOW_LIVE_REPO_TESTS, the same switch git_ops's own
    destructive-git fuse reads), NOT on the DATA opt-in: they are separate switches, and a run
    that opts into live DATA has not opted into reading the live repo's update transaction.
    """
    if os.environ.get("OUROBOROS_ALLOW_LIVE_REPO_TESTS") == "1":
        return
    from supervisor import git_ops

    global _PYTEST_REPO_FALLBACK
    if _PYTEST_DATA_DIR is None and _PYTEST_REPO_FALLBACK is None:
        _PYTEST_REPO_FALLBACK = pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-pytest-repo-"))
    repo_root = (_PYTEST_DATA_DIR or _PYTEST_REPO_FALLBACK) / "repo"
    git_ops.REPO_DIR = repo_root.resolve(strict=False)
    git_ops.REPO_DIR.mkdir(parents=True, exist_ok=True)


def git_ops_repo_root() -> pathlib.Path:
    """The repo root this pytest session binds git_ops (and worker children) to."""
    from supervisor import git_ops

    return git_ops.REPO_DIR


def _bind_pytest_runtime_roots() -> None:
    """Rebind modules that may have been imported before conftest set the env."""
    _bind_pytest_repo_root()
    if _PYTEST_DATA_DIR is None:
        return
    root = _PYTEST_DATA_DIR.resolve(strict=False)
    import ouroboros.config as config
    from supervisor import git_ops, queue, state, workers

    config.DATA_DIR = root
    config.SETTINGS_PATH = root / "settings.json"
    state.init(root, state.TOTAL_BUDGET_LIMIT)
    queue.init(root)
    # git_ops has no env fallback: keep every rescue/log writer on the disposable
    # data root without init(), which would also overwrite branch/remote authority.
    git_ops.DRIVE_ROOT = root
    workers.DRIVE_ROOT = root
    # git_ops.DRIVE_ROOT was the one runtime root this rebind list missed
    # (issue #455): _log_supervisor and the reset/rescue writers resolve
    # supervisor.jsonl through it. Un-pinned it now lazily follows the env
    # (git_ops.__getattr__), but the explicit session pin keeps every writer
    # on ONE root even for tests that mutate OUROBOROS_DATA_DIR mid-test.
    from supervisor import git_ops

    git_ops.DRIVE_ROOT = root
    # spawn_workers hands str(workers.REPO_DIR) to every child, and the child binds git_ops to
    # it — so leaving this at the live default would send workers started BY A TEST back at the
    # operator's checkout, undoing the isolation above.
    workers.REPO_DIR = git_ops_repo_root()


def _mock_pollution_files(root: pathlib.Path) -> set[pathlib.Path]:
    """Mock-named pollution in the repo root.

    Catches both the ``<MagicMock ...>`` repr files AND a literal ``MagicMock``
    directory — the latter is what an unmocked ``ctx.drive_root / ...`` write
    materialises (``MagicMock/mock.drive_root.__truediv__()...``). The earlier
    file-only guard missed the directory form, which then rode a ``git add -A``
    into a release.
    """
    out: set[pathlib.Path] = set()
    try:
        for p in root.iterdir():
            if p.is_file() and "<MagicMock" in p.name:
                out.add(p)
            elif p.is_dir() and (p.name == "MagicMock" or p.name.startswith("<MagicMock")):
                out.add(p)
    except OSError:
        return out
    return out


# Files whose tests spawn REAL OS processes / bind REAL ports / mutate process-global state.
# Under `pytest -n` (xdist) they flake — or crash a worker, which (with --max-worker-restart=0)
# fails that worker's WHOLE co-located batch, surfacing as spurious failures in unrelated files.
# So CI **and the hermetic commit gate** (ouroboros/preflight_runner.py, v6.88.0) run them in a
# SERIAL pass (`-m serial`) and exclude them from the parallel pass (`-m "not serial" -n auto`);
# in the gate a crashed worker is a named hard block, not a retry. A NEW real-process/port/
# global-state test should mark itself `@pytest.mark.serial` (preferred) or be added here.
# See docs/DEVELOPMENT.md "Pytest marker lanes".
_SERIAL_TEST_FILES = frozenset({
    "test_workspace_executor.py",
    # Themed siblings of test_workspace_executor.py; they spawn the same real
    # processes, so the whole family stays in the serial lane.
    "test_workspace_executor_services.py",
    "test_workspace_executor_docker.py",
    "test_workspace_executor_admission.py",
    "test_workspace_executor_cleanup.py",
    "test_process_custody.py",
    "test_kill_process_tree_orphans.py",
    "test_zombie_prevention.py",
    "test_worker_crash_retry.py",
    "test_process_resource_leaks.py",
    "test_restart_reconnect.py",
    # spawns a real pytest subprocess via run_hermetic_pytest + its reaper kills whole process
    # trees / sweeps processes referencing a temp root → can collateral-damage sibling xdist
    # workers under -n (their unrelated tests then fail as a crashed-worker batch).
    "test_preflight_runner.py",
    "test_preflight_process_containment.py",
    # Imports/mutates the process-global server settings facade; when xdist
    # reuses a worker after unrelated server tests, cached route/probe state can
    # escape monkeypatch restoration and turn the mocked capability probe into
    # a real network attempt. Keep the whole hot-reload contract in the serial
    # lane, matching its process-global subject.
    "test_settings_budget_hotreload.py",
    # spawns real long-lived sleeper subprocesses via the legacy ouroboros.tools.services path
    # AND mutates the module-global tools.services._SERVICES (NOT covered by the
    # _isolate_workspace_executor_globals fixture, which isolates a different dict).
    "test_services_tool_v2.py",
    # Its own autouse fixture documents that the writer fence "deliberately latches PROCESS-wide
    # state" (workers admission/survivor/blocker latches, update_merge/git_ops module globals);
    # under -n the replace-family no-side-effect pins (replace_env["calls"] == []) intermittently
    # observe git calls leaked by co-located modules. Same module-global class -> serial lane.
    "test_update_apply_routing.py",
})


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Tag whole-file serial suites with the `serial` marker BEFORE pytest's own `-m`
    deselection runs (tryfirst), so `-m "not serial"` / `-m serial` partition them correctly.
    Tests that carry their own `@pytest.mark.serial` decorator are honored natively too."""
    for item in items:
        if pathlib.Path(str(item.fspath)).name in _SERIAL_TEST_FILES:
            item.add_marker(pytest.mark.serial)


def pytest_sessionstart(session):  # noqa: ARG001
    _bind_pytest_runtime_roots()
    _install_pytest_child_isolation()
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    session.config._ouroboros_initial_mock_pollution = _mock_pollution_files(repo_root)


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    # Under pytest-xdist this hook fires on the controller AND every worker process against the
    # SHARED repo root. Run the repo-root pollution sweep + exitstatus mutation ONLY on the
    # controller (the single authority): otherwise workers race the same shutil.rmtree and each
    # set their own session.exitstatus, manufacturing a non-deterministic failed-shaped run.
    # Workers carry a `workerinput` config attribute; the controller (and any serial run) do not.
    if not hasattr(session.config, "workerinput"):
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        initial = getattr(session.config, "_ouroboros_initial_mock_pollution", set())
        leaked = sorted(_mock_pollution_files(repo_root) - initial)
        if leaked:
            paths = ", ".join(str(p.relative_to(repo_root)) for p in leaked[:5])
            # Clean it so it never rides a git add -A into a commit, THEN fail so the
            # offending test is fixed at its source (an unmocked drive_root/path).
            for p in leaked:
                try:
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        p.unlink(missing_ok=True)
                except OSError:
                    pass
            # Fail the run loudly WITHOUT relying on pytest.Exit (absent in the pinned pytest
            # version → it would crash the session with AttributeError instead of cleanly
            # failing). Setting session.exitstatus marks the run failed; a printed banner names
            # the offending paths so the unmocked drive_root/path is fixed at its source.
            print(
                f"\n\n❌ TEST POLLUTION: mock-named paths leaked into repo root (cleaned): {paths}\n",
                file=sys.stderr,
            )
            session.exitstatus = 1
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:  # xdist worker: hand the leak list to the controller
        workeroutput["thread_leaks"] = list(_THREAD_LEAKS)
    # Per-process temp data dir (unique mkdtemp per controller/worker) — clean on EVERY process.
    if _PYTEST_DATA_DIR is not None:
        shutil.rmtree(_PYTEST_DATA_DIR, ignore_errors=True)
    if _PYTEST_REPO_FALLBACK is not None:
        shutil.rmtree(_PYTEST_REPO_FALLBACK, ignore_errors=True)


def pytest_unconfigure(config):  # noqa: ARG001
    # Keep child isolation active through every session-finish hook; some tests
    # exercise that hook directly before the real pytest session has ended.
    _restore_pytest_child_isolation()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):  # noqa: ARG001
    """Install a fresh asyncio event loop for the test *call* phase.

    Problem: asyncio.run() closes the loop it creates, leaving no current
    loop for the next test's asyncio.get_event_loop() call (RuntimeError).

    This hook installs a fresh loop BEFORE the test body and closes it
    AFTER, preventing cross-test contamination.  The loop is set to None
    after the call phase; a companion pytest_runtest_teardown hook
    installs a temporary loop for fixture finalizers.
    """
    test_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(test_loop)
    # Thread-hygiene baseline, taken AFTER every fixture is set up: a thread a module- or
    # session-scoped fixture starts on its first use (the E2E stub model server) belongs to
    # that fixture for its whole scope and is not a leak of this test; only threads the TEST
    # BODY leaves behind are named at teardown. Thread OBJECTS, not idents: CPython recycles
    # an ident once a baseline thread exits, so a leaked thread could inherit one.
    item.stash[_THREADS_BEFORE_ITEM] = set(threading.enumerate())
    yield  # test body runs here
    test_loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture(autouse=True)
def _rebind_runtime_roots_between_tests():
    _bind_pytest_runtime_roots()
    yield


@pytest.fixture(autouse=True)
def _restore_gateway_settings_bindings_between_tests():
    """``server._sync_gateway_settings_module()`` copies the server module's CURRENT
    ``load_settings`` / ``save_settings`` / ``_apply_settings_to_env`` /
    ``apply_runtime_provider_defaults`` onto ``ouroboros.gateway.settings`` on every
    settings GET/POST, so a test that monkeypatches ``server.load_settings`` and then
    hits the endpoint leaves the TEST-LOCAL loader bound on the gateway module after
    its own monkeypatch is undone (monkeypatch never saw that assignment). The next
    test of the same xdist worker that saves settings through the gateway then reads
    stale "previous rows" and the one-time R12 disclosure fires twice
    (``test_the_save_that_first_makes_the_triad_retrieve_discloses_once_with_numbers``
    after ``test_review_cycles.py``). Snapshot the four bindings before each test and
    restore them afterwards — the same shape as the autouse `_os_environ_isolation`
    environment restore below."""
    try:
        from ouroboros.gateway import settings as _gateway_settings
    except Exception:  # pragma: no cover - the gateway package is always importable in CI
        yield
        return
    names = ("load_settings", "save_settings", "_apply_settings_to_env", "apply_runtime_provider_defaults")
    saved = {name: getattr(_gateway_settings, name, None) for name in names}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                continue
            setattr(_gateway_settings, name, value)


@pytest.fixture(autouse=True)
def _scrub_inherited_subagent_selection(monkeypatch):
    """Keep tests independent of the operator's saved actor list, account pin
    and structured reviewer panel: a test that pins the legacy comma-list
    branch must never read the shell's `OUROBOROS_REVIEWER_SLOTS`."""
    monkeypatch.delenv("OUROBOROS_SUBAGENT_PROFILE", raising=False)
    monkeypatch.delenv("OUROBOROS_SUBAGENTS", raising=False)
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    # The task's absolute ceiling bounds recorded acceptance durations; a shell
    # export must not move the numbers the pacing tests derive from the getter.
    monkeypatch.delenv("OUROBOROS_TASK_ABS_CEILING_SEC", raising=False)


def restored_os_environ():
    """Snapshot os.environ, yield, restore it IN PLACE (clear + update).

    Restoring on the real os._Environ preserves the C-level putenv sync that
    spawned subprocesses inherit from — swapping a plain dict in (the removed
    monkeypatch idiom) severs it. Plain generator so the isolation contract is
    directly testable without pytest plumbing.
    """
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def _os_environ_isolation():
    """Restore the EXACT pre-test os.environ after every test.

    Tests exercise apply_settings_to_env(), owner-settings writers, and ad-hoc
    os.environ mutation — the benchmark launchers write it directly
    (`run_tb.apply_all_model`, `fixed_model_actor_snapshot(target=os.environ)`)
    and `monkeypatch.delenv(raising=False)` records nothing for a key that did
    not exist; under xdist a leaked variable poisons whichever tests share the
    worker afterwards (order-dependent flakes, the `benchmark-scope-1`
    contamination class). One structural snapshot/restore closes the whole leak
    class instead of policing each call site.
    """
    yield from restored_os_environ()


@pytest.fixture(autouse=True)
def _reset_runtime_mode_baseline_between_tests():
    """v5.1.2 iter-2 test isolation fix (Gemini finding F2-7):
    ``ouroboros.config._BOOT_RUNTIME_MODE`` is a module-level global
    pinned by ``initialize_runtime_mode_baseline``. Tests that boot a
    Starlette ``TestClient`` trigger ``server.lifespan`` which pins the
    baseline; subsequent tests inherit the pin and may see different
    rank-comparison behaviour depending on test order. Reset to ``None``
    + remove the env var on every test boundary so each test starts
    with the documented "no pin" state. Tests that need a pin call
    ``initialize_runtime_mode_baseline(...)`` explicitly.
    """
    # The baseline reset only clears OUROBOROS_BOOT_RUNTIME_MODE; the MAIN runtime-mode
    # env (`OUROBOROS_RUNTIME_MODE`, set by apply_settings_to_env/save_settings) is what
    # `get_runtime_mode()` reads.  The operator's inherited runtime mode must not change
    # test semantics either: hermetic review intentionally loads the live non-secret
    # settings before spawning pytest.  Remove it for the test so the documented
    # default applies; the autouse os.environ snapshot restores it afterwards.
    os.environ.pop("OUROBOROS_RUNTIME_MODE", None)
    try:
        from ouroboros.config import reset_runtime_mode_baseline_for_tests
        reset_runtime_mode_baseline_for_tests()
    except Exception:
        pass
    yield
    try:
        from ouroboros.config import reset_runtime_mode_baseline_for_tests
        reset_runtime_mode_baseline_for_tests()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _hide_bundled_skills(monkeypatch):
    """Keep skill tests isolated from the developer machine's data plane.

    v4.50: neutralise the data-plane skills lookup so a developer
    machine with installed skills under ``~/Ouroboros/data/skills/`` does
    not poison test results. ``discover_skills`` consults
    ``_resolve_data_skills_dir`` for its primary scan; pinning that to
    ``None`` forces tests to either pass an explicit ``drive_root`` (the
    new contract since v4.50 — the helper now honours that argument)
    or stick to ``OUROBOROS_SKILLS_REPO_PATH`` fixtures under tmp_path.

    Production keeps the default behaviour untouched; this fixture only
    neutralises global data-plane lookups inside the pytest process.
    """
    # Patch the data-plane resolver to None unless the caller supplied
    # an explicit ``drive_root`` (in which case the v4.50 implementation
    # honours that argument and never touches the global). The signature
    # check via ``*args`` keeps the fixture compatible with both the
    # legacy zero-arg call and the new drive_root-aware one.
    real_resolver = None
    try:
        import ouroboros.skill_loader as loader_mod
        real_resolver = loader_mod._resolve_data_skills_dir
    except Exception:
        pass

    def _hermetic_resolver(*args, **kwargs):
        if args and args[0] is not None:
            return real_resolver(*args, **kwargs) if real_resolver else None
        return None

    if real_resolver is not None:
        monkeypatch.setattr(
            "ouroboros.skill_loader._resolve_data_skills_dir",
            _hermetic_resolver,
        )


@pytest.fixture(autouse=True)
def _isolate_workspace_executor_globals():
    """Isolate process/service registry module-globals between tests (parallel-safety).

    Two modules keep service/process state in module-level dicts that nothing reset between tests
    — a latent ordering bug that pytest-xdist's test REDISTRIBUTION exposes (a test inherits
    another's leftover registry → e.g. the docker-cleanup tests flake under ``-n``):
      * ``ouroboros.workspace_executor._SERVICES`` / ``_FOREGROUND`` (re-entrant ``_STATE_LOCK``);
      * the legacy ``ouroboros.tools.services._SERVICES`` (a PLAIN ``_LOCK``).
    Snapshot → clear → run → restore each around every test so each starts from an empty registry,
    in both serial and parallel runs. Registry isolation ONLY — the records may wrap live Popen
    handles, so we never terminate them (production owns process teardown). Each module is
    lazy-imported under its own guard so a stripped build still collects, and only raw dict ops run
    under the lock (never a services function that re-acquires the plain ``_LOCK`` → no deadlock).
    Makes the ad-hoc manual ``_SERVICES.clear()`` calls in the executor tests redundant (harmless).
    """
    try:
        from ouroboros import workspace_executor as we
    except Exception:
        we = None
    try:
        from ouroboros.tools import services as svc
    except Exception:
        svc = None
    if we is not None:
        with we._STATE_LOCK:
            saved_we_services = dict(we._SERVICES)
            saved_we_foreground = dict(we._FOREGROUND)
            we._SERVICES.clear()
            we._FOREGROUND.clear()
    if svc is not None:
        with svc._LOCK:
            saved_svc_services = dict(svc._SERVICES)
            svc._SERVICES.clear()
    try:
        yield
    finally:
        if we is not None:
            with we._STATE_LOCK:
                we._SERVICES.clear()
                we._SERVICES.update(saved_we_services)
                we._FOREGROUND.clear()
                we._FOREGROUND.update(saved_we_foreground)
        if svc is not None:
            with svc._LOCK:
                svc._SERVICES.clear()
                svc._SERVICES.update(saved_svc_services)


@pytest.fixture(autouse=True)
def _isolate_repo_writer_gate():
    """Reset the process-global repo-writer admission latch between tests.

    ``supervisor.workers._repo_writer_gate_reason`` is process-wide by design (the
    managed-update fence). A test that drives a REAL ``rollback_managed_update``
    boot path closes it with ``reopen_writer_admission=False`` — deliberately, on
    the production contract that a restart clears it — but the pytest process
    never restarts, so the latch leaks into whatever test xdist schedules next
    (e.g. the emergency-cleanup shutdown test then sees ``preserve_pending``).
    Snapshot → run → restore, same pattern as the service-registry isolation."""
    try:
        from supervisor import workers
    except Exception:
        yield
        return
    with workers._repo_writer_gate_lock:
        saved = workers._repo_writer_gate_reason
    try:
        yield
    finally:
        with workers._repo_writer_gate_lock:
            workers._repo_writer_gate_reason = saved


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):  # noqa: ARG001
    """Keep a valid asyncio event loop available during the teardown phase.

    Fixture finalizers run during teardown (LIFO order).  If they call
    asyncio.get_event_loop() after a test that used asyncio.run(), they
    would raise RuntimeError because pytest_runtest_call already cleared
    the loop.  This hook installs a temporary loop for teardown and
    closes it afterwards.
    """
    teardown_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(teardown_loop)
    yield  # fixture finalizers and teardown run here
    teardown_loop.close()
    asyncio.set_event_loop(None)
    _fail_if_the_password_resolver_leaked(item)
    _fail_if_a_thread_leaked(item)


_PRISTINE_PASSWORD_RESOLVER = None


def _fail_if_the_password_resolver_leaked(item):
    """A started-and-never-stopped ``patch("ouroboros.server_auth.get_configured_network_password")``
    on a shared xdist worker made the password gate answer '' for every later module (the rc.11
    macos-latest red, the rc.12 ubuntu/macos red — the victim was named, never the leaker). After
    EVERY fixture of the item is torn down (monkeypatch included) the module attribute must be the
    genuine function again; otherwise the test that leaked it is named here and the attribute is
    restored so no victim fails by worker ordering."""
    import os

    import ouroboros.server_auth as server_auth

    global _PRISTINE_PASSWORD_RESOLVER
    current = server_auth.__dict__.get("get_configured_network_password")
    genuine = (getattr(current, "__module__", None) == "ouroboros.server_auth"
               and getattr(current, "__name__", "") == "get_configured_network_password")
    if genuine:
        _PRISTINE_PASSWORD_RESOLVER = _PRISTINE_PASSWORD_RESOLVER or current
        return
    server_auth.get_configured_network_password = _PRISTINE_PASSWORD_RESOLVER or (
        lambda: server_auth.resolve_network_password(
            os.environ.get(server_auth.NETWORK_PASSWORD_KEY, ""), server_auth.load_settings))
    pytest.fail(f"{item.nodeid} left ouroboros.server_auth.get_configured_network_password patched "
                f"({type(current).__name__}); a started patch was never stopped", pytrace=False)


# ---- thread hygiene: name the test that LEAKS a thread, not the victim it pollutes ----
#
# A daemon thread that outlives its test keeps running on the shared xdist worker: a 0.5 s poll
# loop lands in a later test's GLOBAL ``time.sleep`` patch (tests/test_delegate_hold.py pinned
# its backoff by presence instead of position for that), a settings-to-environment re-applier
# overwrites os.environ after the conftest snapshot restored it (tests/test_server_auth.py was
# rewritten around a pure resolver for that). Both times the victim was named and the leaker
# never was. Same shape as the password-resolver guard above: snapshot the live thread idents
# BEFORE the item's fixtures set up, and after EVERY fixture of the item is torn down every
# thread that appeared since must be gone (a bounded grace lets a stopped-but-not-joined
# thread finish); otherwise the item is failed with the thread names and recorded for the
# session report line.
_THREADS_BEFORE_ITEM = pytest.StashKey()
_THREAD_LEAKS: list = []  # (nodeid, [thread names]) — session-scoped, merged onto the controller
_THREAD_LEAK_GRACE_SEC = 2.0
# By-design detached threads, listed by name prefix — each entry names its owner and why.
_DETACHED_THREAD_NAME_PREFIXES = (
    # ouroboros/project_naming.py: the inner namer call is deliberately abandoned when it
    # overruns the wall-clock bound (the outer ``namer-<task_id>`` returns without joining it).
    "namer-call-",
    # ouroboros/gateway/onboarding.py: the idle worker of the module-lifetime single-worker
    # snapshot executor — kept process-global on purpose so a retried completion JOINS an
    # in-flight daemon read (issue #464) instead of starting a second blocked thread.
    "onboarding-snapshot",
)



def _fail_if_a_thread_leaked(item):
    before = item.stash.get(_THREADS_BEFORE_ITEM, None)
    if before is None:
        return
    deadline = time.monotonic() + _THREAD_LEAK_GRACE_SEC
    leaked = []
    for thread in threading.enumerate():
        if thread in before or thread is threading.current_thread():
            continue
        if thread.name.startswith(_DETACHED_THREAD_NAME_PREFIXES):
            continue
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            leaked.append(f"{thread.name}{'' if thread.daemon else ' (non-daemon)'}")
    if not leaked:
        return
    _THREAD_LEAKS.append((item.nodeid, leaked))
    pytest.fail(f"{item.nodeid} leaked {len(leaked)} thread(s) still alive after every fixture "
                f"was torn down: {', '.join(leaked)} — stop/join it at its owner (a fixture "
                f"finalizer or the test's own missing stop), do not widen the tolerance of the "
                f"test it pollutes", pytrace=False)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):  # noqa: ARG001
    # pytest-xdist controller: merge each worker's leak list (shipped via workeroutput below).
    _THREAD_LEAKS.extend(getattr(node, "workeroutput", {}).get("thread_leaks", []))


def pytest_terminal_summary(terminalreporter):
    if _THREAD_LEAKS:
        tests = ", ".join(f"{nodeid} [{', '.join(names)}]" for nodeid, names in _THREAD_LEAKS)
        terminalreporter.write_line(
            f"thread hygiene: {sum(len(n) for _, n in _THREAD_LEAKS)} leaked thread(s) in "
            f"{len(_THREAD_LEAKS)} test(s): {tests}")
    else:
        terminalreporter.write_line("thread hygiene: no leaked threads")


# Pre-v5.15 conftest exported four fixtures (``make_git_repo``, ``tool_context``,
# ``make_chat_mock``, ``make_extension_skill``) that no test ever requested as a
# parameter. They were removed in v5.15.0; tests build their own minimal repos /
# contexts under ``tmp_path`` because the per-test layouts diverged enough that a
# shared fixture was always wrong (different branch names, different ``ToolContext``
# shapes, ``MagicMock`` vs real, etc.).


