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
import runpy
import subprocess
import sys
import tempfile
import threading
import time
import zlib

_PYTEST_DATA_DIR = None
_PYTEST_ROOT = None
# Empty in an explicit live-DATA run, which installs no disposable defaults.
_PYTEST_DEFAULTS: dict = {}
# The retained short root of scripts/safe_test.py; each pytest controller claims a
# fresh basetemp beneath it in pytest_configure. Empty for bare pytest.
#
# Every pytest process of a run under it — controller, xdist worker, a nested run
# that inherits this environment — creates its session root as a SIBLING directly
# beneath it and redirects its temp directory into that root. Bare pytest keeps the
# invoking temp directory instead: no launcher vouches for a short parent there, and
# a root made beneath an inherited, already redirected TMPDIR would add one level per
# process generation until socket paths pass the AF_UNIX limit (104 bytes on macOS).
_SAFE_TEMP_ROOT = os.environ.get("OUROBOROS_TEST_TEMP_ROOT", "")


# Repo root for a live-DATA run, which has no pytest data dir to hang it off. Created lazily,
# so the hermetic lane never creates an unused temp dir.
_PYTEST_REPO_FALLBACK = None
_NEVER_INHERITED = ("OUROBOROS_MANAGED_BY_LAUNCHER", "OUROBOROS_MANAGED_REPO_DIR",
                    "OUROBOROS_REVIEW_RUN_CAP_USD", "OUROBOROS_CLAUDEXOR_ATTACH_HOME")
if os.environ.get("OUROBOROS_ALLOW_LIVE_DATA_TESTS") != "1":
    _LIVE_DATA_ROOT = (
        os.environ.get("OUROBOROS_TEST_LIVE_DATA_ROOT")
        or os.environ.get("OUROBOROS_DATA_DIR")
        or str(pathlib.Path.home() / "Ouroboros" / "data")
    )
    # Run the stdlib-only helper by path: no package/config import may precede
    # this boundary, including pytest plugins loaded from test modules.
    _PYTEST_ROOT = pathlib.Path(tempfile.mkdtemp(
        prefix="p" if _SAFE_TEMP_ROOT else "ouroboros-pytest-", dir=_SAFE_TEMP_ROOT or None))
    _environment = runpy.run_path(str(pathlib.Path(__file__).resolve().parents[1]
                                    / "ouroboros" / "test_environment.py"))
    _PYTEST_DEFAULTS = _environment["isolated_environment"](
        _PYTEST_ROOT, _PYTEST_ROOT / "data" / "repo", source={},
    )
    if not _SAFE_TEMP_ROOT:
        for _key in ("TMPDIR", "TEMP", "TMP", "PYTEST_DEBUG_TEMPROOT"):
            _PYTEST_DEFAULTS.pop(_key, None)
    # Bare pytest retains explicitly supplied lane/provider controls (integration
    # CI needs them). safe_test.py and preflight scrub the whole owner environment.
    # HOME is already disposable. Leave Deliverables home-derived so tests that
    # select their own HOME/user-files jail do not inherit an unrelated pin.
    _PYTEST_DEFAULTS["OUROBOROS_DELIVERABLES_ROOT"] = ""
    # xdist supplies basetemp under the controller's session tree, alongside (not
    # beneath) each worker's own sibling root. Fence their common parent: the
    # launcher's root under safe_test, the invoking temp directory otherwise.
    _PYTEST_DEFAULTS["GIT_CEILING_DIRECTORIES"] = str(_PYTEST_ROOT.parent)
    os.environ.update(_PYTEST_DEFAULTS)
    # Run identities, never lane controls: launcher authority, and the isolated
    # review's run cap / attach-only engine selection (no test reaches a host engine).
    for _key in _NEVER_INHERITED:
        os.environ.pop(_key, None)
    os.environ["OUROBOROS_TEST_LIVE_DATA_ROOT"] = _LIVE_DATA_ROOT
    _PYTEST_DATA_DIR = pathlib.Path(os.environ["OUROBOROS_DATA_DIR"])
    if _SAFE_TEMP_ROOT:
        # pytest's capture already cached the inherited directory before this ran.
        tempfile.tempdir = os.environ["TMPDIR"]
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]

import pytest
pytest.register_assert_rewrite("tests.ui_media_delivery_smoke")
pytest_plugins = ["tests.browser_lane", "tests.ci_evidence"]


@pytest.fixture
def preflight_timeout_diagnostics(request, monkeypatch, tmp_path):
    """Observe the Windows nested-pytest timeout without changing its gate."""
    import faulthandler
    import json
    from ouroboros import preflight_runner
    from ouroboros.platform_layer import collect_descendant_pids

    trace_dir = tmp_path / "preflight-diagnostics"

    def install(*, dump_after=90):
        trace_dir.mkdir()
        original_probe = preflight_runner._install_worker_probe
        original_kill = preflight_runner._terminate_preflight_tree

        def probe(temp_root):
            module = original_probe(temp_root)
            path = preflight_runner._probe_dir(temp_root) / (module + ".py")
            with path.open("a", encoding="utf-8") as stream:
                stream.write(f'''
import atexit, faulthandler, json, sys, time
_trace = open({str(trace_dir)!r} + "/" + str(os.getpid()) + ".log", "a", encoding="utf-8")
def _trace_event(event, **facts):
    _trace.write(json.dumps(dict(event=event, pid=os.getpid(), ppid=os.getppid(),
        worker=os.environ.get("PYTEST_XDIST_WORKER", "controller"),
        monotonic=time.monotonic(), **facts)) + "\\n")
    _trace.flush()
_trace_event("probe_import", stdout_fd=sys.stdout.fileno(), stderr_fd=sys.stderr.fileno())
faulthandler.dump_traceback_later({dump_after!r}, repeat=True, file=_trace)
def _trace_exit():
    _trace_event("atexit")
    faulthandler.cancel_dump_traceback_later()
atexit.register(_trace_exit)
def pytest_sessionstart(session):
    _trace_event("sessionstart")
def pytest_sessionfinish(session, exitstatus):
    _trace_event("sessionfinish", exitstatus=int(exitstatus))
def pytest_unconfigure(config):
    _trace_event("unconfigure")
def pytest_testnodedown(node, error):
    _trace_event("worker_down", gateway=node.gateway.id, error_type=type(error).__name__)
''')
            return module

        def before_kill(proc, temp_root):
            try:
                facts = {"event": "before_kill", "pid": proc.pid,
                         "monotonic": time.monotonic(), "returncode": proc.poll(),
                         "descendant_pids": collect_descendant_pids(proc.pid)}
                for name in ("stdout", "stderr"):
                    stream = getattr(proc, name)
                    reader = getattr(proc, name + "_thread", None)
                    facts[name] = {"closed": stream.closed if stream else None,
                                   "fd": stream.fileno() if stream and not stream.closed else None,
                                   "reader_alive": reader.is_alive() if reader else None}
                with (trace_dir / "parent.log").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(facts) + "\n")
                    stream.flush()
                    faulthandler.dump_traceback(file=stream)
            except Exception as exc:
                print(f"preflight diagnostic capture failed: {type(exc).__name__}")
            finally:
                original_kill(proc, temp_root)

        monkeypatch.setattr(preflight_runner, "_install_worker_probe", probe)
        monkeypatch.setattr(preflight_runner, "_terminate_preflight_tree", before_kill)
        return trace_dir

    if sys.platform == "win32" and request.node.name == "test_hermetic_pytest_applies_candidate_diff_and_scrubs_live_env":
        install()
    yield install
    if trace_dir.exists():
        for path in sorted(trace_dir.glob("*.log")):
            print(f"\npreflight diagnostic {path.name}:\n{path.read_text(encoding='utf-8')}")

_ORIGINAL_POPEN_INIT = subprocess.Popen.__init__
_PYTEST_CHILD_LIVE_ROOT = os.environ.get("OUROBOROS_TEST_LIVE_DATA_ROOT", "")
_PYTEST_POPEN_PATCHED = False


def _isolated_child_env(value) -> dict:
    child_env = dict(value)
    synthetic_home = (child_env.get("HOME") or child_env.get("USERPROFILE"))
    synthetic_home = synthetic_home and synthetic_home != _PYTEST_DEFAULTS["HOME"]
    home_defaults = {"HOME", "USERPROFILE", "OUROBOROS_APP_ROOT", "OUROBOROS_REPO_DIR",
                     "OUROBOROS_SUBAGENT_PROJECTS_ROOT", "OUROBOROS_SUBAGENT_WORKTREE_ROOT",
                     "OUROBOROS_DELIVERABLES_ROOT", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM"}
    empty_controls = {"PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX", "PYTHONNOUSERSITE"}
    for key, default_value in _PYTEST_DEFAULTS.items():
        if key == "PYTHONDONTWRITEBYTECODE" and key not in child_env and "PYTHONPYCACHEPREFIX" in child_env:
            continue  # Explicit cache selection may intentionally exercise bytecode writes.
        if synthetic_home and key in home_defaults:
            continue  # A test explicitly selected its own synthetic HOME semantics.
        if key not in child_env or (not child_env[key] and key not in empty_controls):
            child_env[key] = default_value
    if not value.get("OUROBOROS_SETTINGS_PATH"):
        child_env["OUROBOROS_SETTINGS_PATH"] = str(
            pathlib.Path(child_env["OUROBOROS_DATA_DIR"]) / "settings.json")
    for key in _NEVER_INHERITED:
        child_env.pop(key, None)
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
        if len(positional) > 10:
            positional[10] = _isolated_child_env(os.environ if positional[10] is None else positional[10])
        else:
            kwargs["env"] = _isolated_child_env(os.environ if kwargs.get("env") is None else kwargs["env"])
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
    _pin_lane_groups(items, config.getoption("--serial-shards"))


_BROWSER_LANE_MARKERS = ("ui_browser", "ui_browser_docker", "browser")


def pytest_collection_finish(session):
    """Browser lanes keep asserting the Dark palette unless a test asks otherwise.

    A client with no saved Appearance choice follows the system, and Playwright's own default
    `prefers-color-scheme` is `light` — so every page a browser test opens would silently render
    Light. Pages and contexts therefore default to `color_scheme="dark"`; a test that wants Light
    passes `color_scheme=` or calls `page.emulate_media(...)`, which still win. Playwright is
    imported only when a browser-lane test was actually collected.
    """
    if not any(item.get_closest_marker(name) for item in session.items for name in _BROWSER_LANE_MARKERS):
        return
    try:
        from playwright.sync_api import Browser
    except Exception:
        return
    for name in ("new_page", "new_context"):
        original = getattr(Browser, name)
        if getattr(original, "_ouroboros_dark_default", False):
            continue

        def dark_by_default(self, *args, __original=original, **kwargs):
            kwargs.setdefault("color_scheme", "dark")
            return __original(self, *args, **kwargs)

        dark_by_default._ouroboros_dark_default = True
        setattr(Browser, name, functools.wraps(original)(dark_by_default))


def pytest_addoption(parser):
    parser.addoption(
        "--serial-shards", type=int, default=0,
        help="With `--dist loadgroup`: pin the serial tests to this many file-sharded xdist groups "
             "so ONE run covers both lanes (scripts/run_tests.py). 0 leaves scheduling untouched.",
    )


def _pin_lane_groups(items, shards: int) -> None:
    """One-run lane scheduling for `scripts/run_tests.py`; inert (shards == 0) for CI and the gate.

    Under `--dist loadgroup` a serial FILE never splits across workers and never runs beside
    another file of its own shard, so the shard count bounds how many serial files run at once.
    Every other test keeps the scope `--dist loadscope` gives the two-pass recipe (file, or class
    when there is one). Serial tests sort first so their long groups are handed out before the queue drains.
    It runs inside the tryfirst hook because xdist reads `xdist_group` in the same hook.
    """
    if shards <= 0:
        return
    for item in items:
        path = item.nodeid.split("::", 1)[0]
        serial = item.get_closest_marker("serial") is not None
        # Serial: by FILE (the isolation unit). Others: the scope `--dist loadscope` uses (class when present).
        group = f"serial{zlib.crc32(path.encode('utf-8')) % shards}" if serial else item.nodeid.rsplit("::", 1)[0]
        item.add_marker(pytest.mark.xdist_group(group))
    items.sort(key=lambda item: item.get_closest_marker("serial") is None)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if _PYTEST_ROOT is not None:
        config.inicfg["cache_dir"] = str(_PYTEST_ROOT / "pytest-cache")
    _guard_test_tree_deletion(config)


def _guard_test_tree_deletion(config) -> None:
    """pytest deletes temp trees on its own; none of it may happen under a live process.

    Tests start servers, workers and git in tmp_path, and a test's exit does not prove
    them gone. The "failed"/"none" retention policies delete tmp_path after passing
    tests, and an explicit --basetemp is emptied before the session starts. pytest's
    OWN numbered basetemp is never used either: its exit hook deletes earlier
    sessions' trees and, with tmp_path_retention_count=0, the current one — whatever
    the policy says — so every controller, bare or not, claims a fresh explicit one.
    """
    policy = config.getini("tmp_path_retention_policy")
    if policy != "all":
        raise pytest.UsageError(
            f"tmp_path_retention_policy={policy!r} would delete test trees whose processes "
            "were never proven gone; this suite requires 'all'")
    if hasattr(config, "workerinput"):
        return  # xdist hands each worker a fresh basetemp beneath the controller's.
    if config.option.basetemp is None:
        config.option.basetemp = _claim_fresh_basetemp(_basetemp_parent())
        return
    basetemp = os.path.abspath(config.option.basetemp)  # Same lexical normalization as pytest.
    if os.path.lexists(basetemp):
        # pytest empties ANY existing explicit basetemp before the session starts. An
        # EMPTY directory is no proof either: it can still be a surviving process's
        # working directory, and emptiness says nothing about that process.
        raise pytest.UsageError(
            f"--basetemp {basetemp} already exists, and pytest would delete it "
            "before its processes are proven gone; pass a fresh, never-used path")


def _basetemp_parent() -> pathlib.Path:
    """The launcher's root, else this process's own fresh session root.

    A live-DATA run has no session root; it gets a fresh directory of its own.
    """
    if _SAFE_TEMP_ROOT:
        return pathlib.Path(_SAFE_TEMP_ROOT)
    if _PYTEST_ROOT is not None:
        return _PYTEST_ROOT
    return pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-pytest-"))


def _claim_fresh_basetemp(root: pathlib.Path) -> str:
    """One never-used basetemp per pytest invocation, short enough for socket paths.

    The exclusive mkdir of ``b<N>`` is the claim: a later or concurrent session
    (run_tests.py's second pass, a nested run) takes the next number. pytest gets
    a NOT-yet-existing child, so it never has an existing tree to empty.
    """
    for index in range(10_000):
        claim = root / f"b{index}"
        try:
            claim.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return str(claim / "t")
    raise pytest.UsageError(f"no free basetemp claim beneath {root}")


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
    # The per-process session tree (unique mkdtemp per controller/worker) is RETAINED,
    # never deleted here: a finished session does not prove that every process its
    # tests started is gone, and nothing at this layer can prove it. The terminal
    # summary names the tree; cleanup belongs to whoever later proves the run idle.


def pytest_unconfigure(config):  # noqa: ARG001
    # Keep child isolation active through every session-finish hook; some tests
    # exercise that hook directly before the real pytest session has ended.
    _restore_pytest_child_isolation()


_PHASE_EVENT_LOOPS = pytest.StashKey()


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):  # noqa: ARG001
    """Create both phase loops before fixtures can guard socket operations.

    Windows loop construction opens a local socket pair. Network guards must
    remain active throughout the test and its finalizers, so only construction
    precedes fixture setup. This owner closes both loops even if a phase fails.
    """
    loops = []
    try:
        loops.append(asyncio.new_event_loop())
        loops.append(asyncio.new_event_loop())
        item.stash[_PHASE_EVENT_LOOPS] = loops
        yield
    finally:
        for loop in loops:
            loop.close()
        asyncio.set_event_loop(None)
        if _PHASE_EVENT_LOOPS in item.stash:
            del item.stash[_PHASE_EVENT_LOOPS]


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
    test_loop = item.stash[_PHASE_EVENT_LOOPS][0]
    asyncio.set_event_loop(test_loop)
    # Thread-hygiene baseline, taken AFTER every fixture is set up: a thread a module- or
    # session-scoped fixture starts on its first use (the E2E stub model server) belongs to
    # that fixture for its whole scope and is not a leak of this test; only threads the TEST
    # BODY leaves behind are named at teardown. Thread OBJECTS, not idents: CPython recycles
    # an ident once a baseline thread exits, so a leaked thread could inherit one.
    item.stash[_THREADS_BEFORE_ITEM] = set(threading.enumerate())
    try:
        yield  # test body runs here
    finally:
        test_loop.close()
        asyncio.set_event_loop(None)


@pytest.fixture(autouse=True)
def _rebind_runtime_roots_between_tests():
    _bind_pytest_runtime_roots()
    yield


@pytest.fixture(autouse=True)
def _isolate_restart_generation(monkeypatch):
    """Panic and parent observation belong to one process generation, not later tests."""
    from ouroboros import delegate_recovery, server_control

    monkeypatch.setattr(server_control, "_restart_successors", [])
    monkeypatch.setattr(server_control, "_restart_stop_requested", False)
    monkeypatch.setattr(delegate_recovery, "_restart_parent", None)


@pytest.fixture(autouse=True)
def _reset_custody_memo_between_tests():
    """Isolate both custody caches: the row memo is keyed by events-log path,
    while active custody is keyed only by run ID. Tests reuse both identities (and so the
    delegated-activity memo): no consumed prefix, first-wins binding or shown cursor may leak."""
    from ouroboros import delegate_activity, delegate_custody
    from ouroboros.delegate_custody_memo import reset_custody_memo

    delegate_custody._CUSTODY.clear()
    reset_custody_memo()
    delegate_activity.reset_process_memo()
    yield
    delegate_custody._CUSTODY.clear()
    reset_custody_memo()
    delegate_activity.reset_process_memo()


@pytest.fixture(autouse=True)
def _reset_accepted_ids_between_tests():
    """The named-ingress index (``message_ingress._AcceptedIds``) is keyed by chat-log path:
    no folded prefix outlives its test, whatever the next one writes at that path."""
    from supervisor.message_ingress import reset_accepted_ids

    reset_accepted_ids()
    yield
    reset_accepted_ids()


@pytest.fixture(autouse=True)
def _unlatch_supervisor_event_bus_between_tests():
    """A TestClient lifespan runs the server shutdown, whose ``workers.shutdown_event_q()``
    latches ``_EVENT_Q_SHUTDOWN`` for the rest of the xdist worker; the next test in that
    worker that publishes on the bus (``kill_workers_for_update``, a promote, a wake) then
    raises "supervisor event bus is shutting down" against a fixture it never saw. Several
    modules already unlatch it locally (test_promote_event_transport, test_inflight_indicator_seams);
    this does it once for every test. A test that wants the latch sets it itself (monkeypatch)."""
    from supervisor import workers

    workers._EVENT_Q_SHUTDOWN = False
    yield


@pytest.fixture(autouse=True)
def _clear_server_stop_flags_between_tests():
    """The same class as the event-bus latch above: a TestClient lifespan teardown (and the
    shutdown tests) SET the process-global ``_supervisor_stop`` / ``_restart_requested`` events
    and nothing clears them, so every later test in that xdist worker that ran the off-thread
    custody pass saw "this process is stopping" and the pass ended before its first step — red
    only in the full battery. A test that wants a flag sets it itself."""
    import sys

    server_process = sys.modules.get("ouroboros.server_process")
    if server_process is not None:
        server_process._supervisor_stop.clear()
        server_process._restart_requested.clear()
    yield


@pytest.fixture(autouse=True)
def _keep_process_logging_out_of_the_pytest_process(monkeypatch):
    """Restart and shutdown tests run ``server.main()`` in this process and launcher tests import
    ``launcher``; each would run the per-process logging bootstrap here, leaving root handlers
    bound to that test's tmp dir and captured stderr, the root level at INFO and both excepthooks
    replaced for the rest of the xdist worker. A later test then logged through the stale handlers
    (the supervisor-watchdog stack test lost its watchdog thread that way, red only in the full
    battery). The pytest process owns its own logging: the bootstrap is a no-op here and both hooks
    are restored after every test; tests/test_process_logging.py runs the bootstrap in fresh
    interpreters."""
    import sys
    import threading

    from ouroboros import process_logging

    monkeypatch.setattr(process_logging, "_configured", True)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)


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
def _isolate_evolution_stop_latch(monkeypatch):
    """A received evolution Stop is a process-lifetime latch (#1307); no test inherits one."""
    from supervisor import evolution_lifecycle

    monkeypatch.setitem(evolution_lifecycle._STOP_LATCH, "stopped", False)


@pytest.fixture(autouse=True)
def _isolate_workspace_executor_globals():
    """Snapshot/reset/restore service registries AND their process-lifetime Panic latches.

    Real Panic requests retire admission even with empty registries. A mocked hard exit in
    test_post_task_evolution left that latch set for the next black-box service test.
    Each test gets fresh admission; Panic still latches for its whole test.
    Never terminate saved Popen handles here: production owns process teardown. Lazy imports
    keep stripped builds collectable; only raw state operations run under either module lock
    (services._LOCK is non-reentrant, so calling a service helper there would deadlock).
    """
    try:
        from ouroboros import owned_shutdown, workspace_executor as we
        owned_shutdown._GENERATION_STOP = owned_shutdown._Stop()  # the one owned-work stop is per process
    except Exception:
        we = None
    try:
        from ouroboros.tools import services as svc
    except Exception:
        svc = None
    if we is not None:
        with we._STATE_LOCK:
            saved_we_panic = we._panic_requested
            we._panic_requested = False
            saved_we_services = dict(we._SERVICES)
            saved_we_foreground = dict(we._FOREGROUND)
            we._SERVICES.clear()
            we._FOREGROUND.clear()
    if svc is not None:
        with svc._LOCK:
            saved_svc_panic = svc._panic_requested
            svc._panic_requested = False
            saved_svc_services = dict(svc._SERVICES)
            svc._SERVICES.clear()
    try:
        yield
    finally:
        if we is not None:
            with we._STATE_LOCK:
                we._panic_requested = saved_we_panic
                we._SERVICES.clear()
                we._SERVICES.update(saved_we_services)
                we._FOREGROUND.clear()
                we._FOREGROUND.update(saved_we_foreground)
        if svc is not None:
            with svc._LOCK:
                svc._panic_requested = saved_svc_panic
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
    teardown_loop = item.stash[_PHASE_EVENT_LOOPS][1]
    asyncio.set_event_loop(teardown_loop)
    try:
        yield  # fixture finalizers and teardown run here
    finally:
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
    retained = [str(path) for path in (_PYTEST_ROOT, _PYTEST_REPO_FALLBACK) if path is not None]
    basetemp = getattr(getattr(terminalreporter.config, "_tmp_path_factory", None), "_basetemp", None)
    if basetemp is not None and not any(pathlib.Path(basetemp).is_relative_to(path) for path in retained):
        retained.append(str(basetemp))
    if retained:
        terminalreporter.write_line("test session trees retained (never deleted in-session): "
                                    + ", ".join(retained))


# Pre-v5.15 conftest exported four fixtures (``make_git_repo``, ``tool_context``,
# ``make_chat_mock``, ``make_extension_skill``) that no test ever requested as a
# parameter. They were removed in v5.15.0; tests build their own minimal repos /
# contexts under ``tmp_path`` because the per-test layouts diverged enough that a
# shared fixture was always wrong (different branch names, different ``ToolContext``
# shapes, ``MagicMock`` vs real, etc.).




@pytest.fixture(autouse=True)
def _isolate_direct_activities(monkeypatch):
    from supervisor import active_activity

    monkeypatch.setattr(active_activity, "_DIRECT_ACTIVITY_REGISTRY", active_activity.DirectActivityRegistry())
