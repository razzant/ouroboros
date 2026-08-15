"""A test that spawns processes, binds sockets or races threads belongs in the serial lane.

The lane split was maintained by hand: someone had to notice that a new test spawned
a real process and remember to add it. Two did not get noticed —
`test_admission_invariants.py` (two real threads racing queue admission plus real
`git` subprocesses per fixture) and the `test_execd_state.py` custodian pair (a real
custodian loop in a thread, `time.sleep`, then `assert thread.is_alive()`). Both ran
in the `-n auto` parallel pass, where a loaded worker can invalidate exactly the
timing claim they assert.

So the contract is enforced by discovery instead of memory: this scan finds every
candidate itself and requires each one to be CLASSIFIED — in the serial lane, in
another lane, or in the exemption list below with a reason. A new process/port/thread
test fails here until someone decides which it is.

Why an exemption list rather than "all of them must be serial": xdist workers are
separate PROCESSES, so a test that starts in-process threads to prove a lock
serializes or a counter does not drop writes is genuinely parallel-safe. Forcing
those into the serial lane would slow the suite and, worse, teach people that the
marker is noise. What is NOT parallel-safe is a claim about wall-clock timing or a
resource shared across processes: a real child process, a real socket, a sleep the
assertion depends on.

The heuristic is a candidate finder, not an oracle. It reads source, so a subprocess
spawned by the CODE UNDER TEST rather than by the test body is invisible to it — the
exemption list is where that judgement gets recorded, and the exact-set assertions
below stop the list from drifting in either direction. One real example of that blind
spot, found by running the full parallel pass rather than by this scan:
`test_execd_spool.py::test_real_process_overrunning_the_spool_quota_is_terminated_and_fully_sealed`
launches its flooder through `_run_process` and asserts the flood outran the quota,
which a loaded worker can invert; it carries its own `serial` marker with that
reasoning written above it.

BOUNDARY — the `subprocess.run` family is NOT detected, deliberately.
`_PROCESS_SIGNALS` names `subprocess.Popen` and the three `socket` constructors, so a
test that shells out through `subprocess.run`, `subprocess.call`, `check_output`,
`check_call` or `os.system` produces no signal at all and is never asked to classify
itself. That is a real hole in the letter of the rule: those calls do fork a real
child, and a planted `subprocess.run([sys.executable, ...])` passes this gate.

It stays open on purpose, and the reason is arithmetic rather than principle. Nearly
every one of those call sites is a short-lived hermetic helper inside a `tmp_path` —
`git init`, `git commit`, one `grep`, a clean-subprocess import smoke — finishing in
milliseconds, sharing nothing across processes and asserting nothing about the clock.
Adding the family to `_PROCESS_SIGNALS` would move ~114 tests across 33 files, almost
all of them older than this feature and none of them suspected, out of the `-n auto`
pass into the serial one, lengthening CI for the whole project to reclassify code that
was never the problem. What this gate exists to catch is a LONG-LIVED child, a bound
port, and an assertion that depends on wall-clock time; `Popen` and `socket` are the
shapes those actually take, which is why the detector is aimed at them.

So a `subprocess.run`-family test is classified by JUDGEMENT, not by this scan. The
RWS v2 tests that genuinely spawn long-lived children, bind loopback ports, drive
Docker/OpenSSH or reason about descriptor numbers were classified by hand and are in
the serial lane: `test_remote_broker_lifecycle.py`, `test_remote_browser_forward.py`,
`test_remote_panic_descriptors.py`, `test_remote_task_session_wiring.py` and
`test_admission_invariants.py` whole-file via `tests/conftest.py::_SERIAL_TEST_FILES`,
plus per-test `serial` markers in `test_execd_spool.py`, `test_execd_state.py`,
`test_remote_workspace_ssh.py`, `test_docker_executor_real_container.py` and
`test_dispatch_prepare.py`. If you add a test that shells out to something
long-running, this file will not tell you — decide it yourself.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

TESTS_DIR = pathlib.Path(__file__).resolve().parent

# Real cross-process resources: a child process or a socket.
_PROCESS_SIGNALS = frozenset({
    "subprocess.Popen", "Popen",
    "socket.socket", "socket.create_server", "socket.create_connection",
})
# In-process concurrency and wall-clock dependence.
_THREAD_SIGNALS = frozenset({"threading.Thread", "Thread"})
# A sleep this long is almost always load-bearing for an assertion that follows it.
_SLEEP_THRESHOLD_SEC = 0.25

# Lanes that are already excluded from the `-n auto` parallel pass.
_OTHER_LANES = frozenset({
    "integration", "browser", "ui_browser", "ui_browser_docker",
    "portable_detail", "skill_smoke",
})

# Candidates deliberately left in the PARALLEL lane, each with the reason it is safe
# there. Every entry is (file, test). The assertions below require this set to match
# reality exactly, so a removed test or a newly written candidate both fail loudly.
PARALLEL_SAFE_CONCURRENCY_TESTS: frozenset[tuple[str, str]] = frozenset({
    # All of these start threads INSIDE the test process to prove a lock serializes,
    # a counter does not drop writes, or a reader/writer pair cannot deadlock. They
    # coordinate with Events and joins rather than with sleeps, touch no port and no
    # child process, and an xdist worker is its own process — so a busy sibling
    # changes their speed, never their outcome.
    ("test_capability_evidence.py", "test_token_density_is_concurrency_safe_under_bench_like_parallelism"),
    ("test_execd_spool.py", "test_concurrent_stream_writers_cannot_oversubscribe_the_host_quota"),
    ("test_extension_isolated_deps.py", "test_isolated_python_deps_do_not_leak_during_overlapping_handlers"),
    ("test_extension_loader.py", "test_concurrent_reconcile_converges_to_one_live_extension"),
    ("test_extension_loader.py", "test_get_settings_rechecks_runtime_close_after_reader_returns"),
    ("test_extension_loader.py", "test_unload_does_not_deadlock_with_inflight_get_settings"),
    ("test_improvement_backlog.py", "test_append_concurrent_writers_do_not_drop_entries"),
    ("test_mcp_client.py", "test_run_async_works_from_sync_caller"),
    ("test_model_concurrency.py", "test_cap_serializes_concurrent_calls"),
    ("test_model_concurrency.py", "test_deadline_failsoft_does_not_block"),
    ("test_usage_accounting.py", "test_legacy_state_projection_cannot_regress_under_reordered_writers"),
    # The `time.sleep(0.3)` in these three is VESTIGIAL: the call under test
    # (`agent_task_pipeline._run_scratchpad_consolidation`) is synchronous, so the
    # mock assertion already holds when the sleep starts. Nothing waits on the
    # clock, so a loaded worker cannot change the outcome — it only pays 0.3s.
    ("test_budget_tracking.py", "test_update_budget_called_after_scratchpad_consolidation"),
    ("test_budget_tracking.py", "test_update_budget_called_after_consolidation"),
    ("test_budget_tracking.py", "test_no_budget_call_when_consolidation_returns_none"),
    # These sleep past a window and then assert that NOTHING happened (a delayed
    # callback did not re-register after unload). Load makes the real window longer
    # than the sleep, which is the safe direction for a negative assertion; a busy
    # sibling cannot make a rejected registration succeed.
    ("test_extension_loader.py", "test_on_unload_delayed_callback_cannot_reregister_surfaces"),
    ("test_extension_loader.py", "test_delayed_post_load_registration_is_rejected"),
    # Rebase onto v6.82: upstream's own concurrency tests, judged one by one against
    # the same rule as the block above. All of them start threads or asyncio tasks
    # INSIDE the test process and coordinate with Events plus bounded joins — no
    # `sleep`, no subprocess, no port. Verified mechanically: each body contains zero
    # `time.sleep`/`asyncio.sleep` calls and zero `subprocess` calls, so a busy xdist
    # sibling changes their speed and never their outcome.
    ("test_cancel_cascade_v664.py", "test_concurrent_cascades_on_overlapping_trees_both_settle"),
    ("test_cancel_cascade_v664.py", "test_overlapping_cascade_cannot_confirm_success_while_the_child_kill_is_in_flight"),
    ("test_evolution_state_integrity_v3.py", "test_boot_reconcile_cannot_resurrect_owner_stopped_campaign"),
    ("test_evolution_state_integrity_v3.py", "test_commit_receipt_uses_campaign_sidecar_before_rescue"),
    ("test_evolution_state_integrity_v3.py", "test_rescue_link_uses_shared_campaign_cas_and_preserves_commit_receipt"),
    ("test_evolution_state_integrity_v3.py", "test_terminal_write_serializes_concurrent_campaign_pause"),
    ("test_telegram_miniapp_lifecycle.py", "test_shutdown_during_prior_owner_reconcile_cleans_up_prior_manager"),
    ("test_telegram_miniapp_lifecycle.py", "test_sidecar_runtime_crash_retries_beyond_host_restart_limit"),
    ("test_telegram_miniapp_lifecycle.py", "test_sidecar_start_retries_beyond_host_restart_limit"),
    # Rebase onto v6.100: judged one by one against the same rule. Both start threads
    # INSIDE the test process and coordinate with an Event/Barrier plus bounded joins —
    # no `sleep`, no subprocess, no port — so a busy xdist sibling changes their speed
    # and never their outcome. The four other candidates this rebase surfaced are NOT
    # here: two bind a real socket and spawn a real child, one runs a real AppRun
    # subprocess, and one uses wall-clock sleeps to CONSTRUCT its race (under load the
    # race would simply not form and the test would pass vacuously). All four are
    # marked serial instead.
    ("test_cancel_intents_phase_a.py", "test_two_concurrent_custodies_on_a_pending_task_settle_exactly_once"),
    ("test_scope_review.py", "test_concurrent_resolution_of_one_route_shares_one_probe"),
})


def _serial_whole_files() -> frozenset[str]:
    """Read `_SERIAL_TEST_FILES` out of conftest by AST, without importing it."""

    tree = ast.parse((TESTS_DIR / "conftest.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(target, "id", "") == "_SERIAL_TEST_FILES" for target in node.targets):
            continue
        call = node.value
        elements = call.args[0].elts if isinstance(call, ast.Call) else []
        return frozenset(
            item.value for item in elements
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        )
    raise AssertionError("tests/conftest.py no longer declares _SERIAL_TEST_FILES")


def _called_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        base = func.value.id if isinstance(func.value, ast.Name) else ""
        return f"{base}.{func.attr}" if base else func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _decorator_marks(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    marks: set[str] = set()
    for decorator in node.decorator_list:
        current = decorator.func if isinstance(decorator, ast.Call) else decorator
        parts: list[str] = []
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        if len(parts) >= 3 and parts[:2] == ["pytest", "mark"]:
            marks.add(parts[2])
    return marks


def _module_marks(tree: ast.Module) -> set[str]:
    marks: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(target, "id", "") == "pytestmark" for target in node.targets):
            continue
        for inner in ast.walk(node.value):
            if isinstance(inner, ast.Attribute):
                marks.add(inner.attr)
    return marks


def _signals(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    found: set[str] = set()
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        name = _called_name(inner.func)
        if name in _PROCESS_SIGNALS:
            found.add(f"process:{name}")
        if name in _THREAD_SIGNALS:
            found.add("thread:threading.Thread")
        if name in {"time.sleep", "sleep"}:
            for arg in inner.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, (int, float))
                    and not isinstance(arg.value, bool)
                    and arg.value >= _SLEEP_THRESHOLD_SEC
                ):
                    found.add(f"sleep:{arg.value}")
    return found


def _candidates() -> list[tuple[str, str, list[str], bool]]:
    """Every signal-bearing test: (file, test, signals, already_in_a_lane)."""

    serial_files = _serial_whole_files()
    out: list[tuple[str, str, list[str], bool]] = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        file_marks = _module_marks(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            signals = _signals(node)
            if not signals:
                continue
            marks = _decorator_marks(node) | file_marks
            in_lane = (
                path.name in serial_files
                or "serial" in marks
                or bool(marks & _OTHER_LANES)
            )
            out.append((path.name, node.name, sorted(signals), in_lane))
    return out


def test_the_scan_still_finds_something():
    """A heuristic that matches nothing would pass forever while proving nothing."""

    candidates = _candidates()
    assert len(candidates) >= 20, len(candidates)
    assert any(row[3] for row in candidates), "no candidate is in a lane — scan broken"


def test_every_process_or_socket_test_is_out_of_the_parallel_lane():
    """The hard half of the rule: a real child process or socket has no exemption.

    These share resources ACROSS processes, so an xdist sibling is not merely slower
    company — it can take the resource, or be taken for the test's own child.
    """

    offenders = [
        f"{file}::{test} {signals}"
        for file, test, signals, in_lane in _candidates()
        if not in_lane and any(signal.startswith("process:") for signal in signals)
    ]
    assert not offenders, (
        "these tests spawn a real subprocess or socket in the PARALLEL lane; add "
        "@pytest.mark.serial (or the file to _SERIAL_TEST_FILES in tests/conftest.py):\n"
        + "\n".join(f"  {row}" for row in offenders)
    )


def test_every_thread_or_timing_test_is_classified():
    """The soft half: a candidate may stay parallel, but only on the record.

    `PARALLEL_SAFE_CONCURRENCY_TESTS` is checked for EXACT equality, so this fails
    when a new candidate appears (classify it) and also when an entry goes stale
    (a renamed or deleted test, or one that moved into the serial lane) — a list of
    exemptions nobody prunes stops describing the suite.
    """

    unclassified = {
        (file, test)
        for file, test, _signals, in_lane in _candidates()
        if not in_lane
    }
    missing = sorted(unclassified - PARALLEL_SAFE_CONCURRENCY_TESTS)
    assert not missing, (
        "these tests start threads or depend on wall-clock timing in the PARALLEL "
        "lane and are not classified. Either mark them serial, or add them to "
        "PARALLEL_SAFE_CONCURRENCY_TESTS with the reason they are safe:\n"
        + "\n".join(f"  {file}::{test}" for file, test in missing)
    )
    stale = sorted(PARALLEL_SAFE_CONCURRENCY_TESTS - unclassified)
    assert not stale, (
        "these exemptions no longer describe a parallel-lane candidate (renamed, "
        "deleted, or already serial) and must be removed:\n"
        + "\n".join(f"  {file}::{test}" for file, test in stale)
    )


@pytest.mark.parametrize(
    "file_name",
    ["test_admission_invariants.py", "test_execd_state.py"],
)
def test_the_two_files_this_contract_was_written_for_are_serial(file_name):
    """Regression: the specific escapes that motivated the scan stay covered."""

    serial_files = _serial_whole_files()
    if file_name in serial_files:
        return
    tree = ast.parse((TESTS_DIR / file_name).read_text(encoding="utf-8"))
    marked = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "serial" in _decorator_marks(node)
    }
    assert "test_custodian_survives_empty_expired_lease_until_explicit_close" in marked
    assert "test_replacement_custodian_waits_for_previous_identity_to_close" in marked


def test_the_scan_states_the_subprocess_run_boundary():
    """The hole must be NAMED in the docstring, not left to be discovered.

    A gate that looks total and is not is worse than one that says where it stops:
    the next contributor reads the exact-set assertions above, concludes the suite is
    fully classified, and never thinks about the `subprocess.run` they just added.
    This fails if the admission is ever quietly deleted while the hole remains.
    """

    doc = __doc__ or ""
    assert "BOUNDARY" in doc
    assert "subprocess.run" in doc
    assert "JUDGEMENT" in doc


@pytest.mark.parametrize(
    "source",
    [
        "def test_x():\n    subprocess.run([sys.executable, '-c', 'import time; time.sleep(5)'])\n",
        "def test_x():\n    subprocess.check_output(['sleep', '5'])\n",
        "def test_x():\n    os.system('sleep 5')\n",
    ],
)
def test_the_named_blind_spot_is_really_blind(source):
    """Pins the admission to reality.

    If one of these starts producing a signal, the docstring's BOUNDARY note is stale
    — and so is the ~114-tests-across-33-files reasoning for leaving it open. Narrow
    the note and re-run the count before widening the detector.
    """

    assert _signals(ast.parse(source).body[0]) == set()


@pytest.mark.parametrize(
    "source",
    [
        "def test_x():\n    subprocess.Popen(['sleep', '5'])\n",
        "def test_x():\n    socket.socket()\n",
        "def test_x():\n    socket.create_server(('127.0.0.1', 0))\n",
    ],
)
def test_the_shapes_the_detector_is_aimed_at_do_signal(source):
    """The other side of the boundary: a long-lived child or a port is caught."""

    assert any(
        signal.startswith("process:") for signal in _signals(ast.parse(source).body[0])
    )


def test_our_process_spawning_suites_are_classified_by_hand():
    """The compensating control for the boundary above, asserted rather than promised.

    These are the RWS v2 suites that really do spawn long-lived children, bind
    loopback ports, drive Docker/OpenSSH or reason about descriptor numbers. The scan
    catches most of them on its own; this pins the CLASSIFICATION so that a later
    edit which, say, replaces a `Popen` with a `subprocess.run` cannot silently drop
    the file back into the parallel lane.
    """

    serial_files = _serial_whole_files()
    for file_name in (
        "test_remote_broker_lifecycle.py",
        "test_remote_browser_forward.py",
        "test_remote_panic_descriptors.py",
        "test_remote_task_session_wiring.py",
        "test_admission_invariants.py",
    ):
        assert (TESTS_DIR / file_name).is_file(), file_name
        assert file_name in serial_files, f"{file_name} left the serial lane"

    for file_name in (
        "test_execd_spool.py",
        "test_execd_state.py",
        "test_remote_workspace_ssh.py",
    ):
        path = TESTS_DIR / file_name
        assert path.is_file(), file_name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        marked = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and "serial" in (_decorator_marks(node) | _module_marks(tree))
        }
        assert marked or file_name in serial_files, (
            f"{file_name} spawns real processes and carries no serial classification"
        )
