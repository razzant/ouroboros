# tests/conftest.py — shared pytest fixtures for the Ouroboros test suite.
#
# Loaded automatically by pytest before any test module runs.
# Cross-module helpers that are not pytest fixtures (e.g. SDK mock, extension
# runtime cleanup) live in ``tests/_shared.py`` instead.
import asyncio
import os
import pathlib
import shutil
import sys
import tempfile

import pytest


_PYTEST_DATA_DIR = None
if os.environ.get("OUROBOROS_ALLOW_LIVE_DATA_TESTS") != "1":
    _PYTEST_DATA_DIR = pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-pytest-data-"))
    os.environ["OUROBOROS_DATA_DIR"] = str(_PYTEST_DATA_DIR)
    os.environ["OUROBOROS_SETTINGS_PATH"] = str(_PYTEST_DATA_DIR / "settings.json")
    # Conftest-WIDE bench-runs isolation. devtools benchmark tests invoke
    # run_*.main(), whose run_root() defaults to the real <repo>/../bench_runs
    # when OUROBOROS_BENCH_RUNS_ROOT is unset — leaking timestamped run dirs and
    # ouroboros_task_body.json stubs into the operator's bench_runs/ (the
    # programbench/swe_bench_pro pollution). A file-local autouse fixture only
    # covered one module; pinning it here covers every test.
    os.environ["OUROBOROS_BENCH_RUNS_ROOT"] = str(_PYTEST_DATA_DIR / "bench_runs")


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
# So CI runs them in a SERIAL pass (`-m serial`) and excludes them from the parallel pass
# (`-m "not serial" -n auto`). A NEW real-process/port/global-state test should mark itself
# `@pytest.mark.serial` (preferred) or be added here. See docs/DEVELOPMENT.md "Pytest marker lanes".
_SERIAL_TEST_FILES = frozenset({
    "test_workspace_executor.py",
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
    # spawns real long-lived sleeper subprocesses via the legacy ouroboros.tools.services path
    # AND mutates the module-global tools.services._SERVICES (NOT covered by the
    # _isolate_workspace_executor_globals fixture, which isolates a different dict).
    "test_services_tool_v2.py",
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
    # Per-process temp data dir (unique mkdtemp per controller/worker) — clean on EVERY process.
    if _PYTEST_DATA_DIR is not None:
        shutil.rmtree(_PYTEST_DATA_DIR, ignore_errors=True)


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
    yield  # test body runs here
    test_loop.close()
    asyncio.set_event_loop(None)


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


# Pre-v5.15 conftest exported four fixtures (``make_git_repo``, ``tool_context``,
# ``make_chat_mock``, ``make_extension_skill``) that no test ever requested as a
# parameter. They were removed in v5.15.0; tests build their own minimal repos /
# contexts under ``tmp_path`` because the per-test layouts diverged enough that a
# shared fixture was always wrong (different branch names, different ``ToolContext``
# shapes, ``MagicMock`` vs real, etc.).
