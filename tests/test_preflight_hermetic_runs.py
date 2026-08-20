"""The real hermetic pytest passes: what actually runs, and what a candidate cannot fake.

Split verbatim out of ``tests/test_preflight_runner.py`` by theme. This module owns the
real two-pass execution and its partition, the parallel pass that really starts more than
one worker, the candidate that can neither switch the parallel plugins off nor fake the
flags, the timeouts and the excerpts they keep, and the child that may not leak into the
next pass.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import time

import pytest

from ouroboros.platform_layer import force_kill_pid, pid_is_alive

from tests._preflight_runner_shared import (
    _PREFLIGHT_PLUGIN_PROBLEMS,
    _REAL_SPAWN_SKIP_REASON,
    _make_repo,
)
from tests._preflight_runner_shared import stub_passes as _stub_passes
from tests._preflight_runner_shared import two_pass_env as _two_pass_env

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
stub_passes = _stub_passes
two_pass_env = _two_pass_env


requires_preflight_plugins = pytest.mark.skipif(
    bool(_PREFLIGHT_PLUGIN_PROBLEMS), reason=_REAL_SPAWN_SKIP_REASON
)

def test_a_timed_out_pass_reports_what_the_killed_child_had_already_flushed(tmp_path, two_pass_env, stub_passes):
    """The serial pass carries no per-test timeout, so a serial hang is exactly
    the case with no other evidence of WHICH test hung — and the post-kill
    `communicate` already collects that evidence. Discarding it left the operator
    with "the serial pass timed out" and nothing else."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    stub_passes([(None, "tests/test_a.py .\ntests/test_hangs_here.py ")])
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is not None
    assert "timed out" in result
    assert "test_hangs_here" in result, "the killed pass's own output was collected then thrown away"

def test_a_second_timeout_keeps_the_excerpt_the_first_one_already_carried(tmp_path, monkeypatch):
    """The retry `communicate` is a bonus source of evidence, not the only one.

    On timeout the pass kills the tree and calls `communicate(timeout=10)` again
    to collect what pytest flushed. That retry can itself time out — an escaped
    grandchild holding the inherited pipe open is the exact case the code
    anticipates — and seeding the excerpt as `""` meant the diagnosis then carried
    nothing at all, losing the last-test evidence it exists to preserve. The FIRST
    `TimeoutExpired` already carries that output (as raw bytes, since
    `communicate` joins the partial reads before applying text-mode decoding), so
    it is the seed and the retry may only enrich it.

    The container is also reaped BEFORE the retry, not only after: the descendant
    that would make the retry hang is precisely the one the container can kill.
    """
    from ouroboros import preflight_runner as pr, process_containment

    order: list = []

    class _StuckProc:
        pid = 4242
        returncode = None
        stdout = None
        stderr = None

        def communicate(self, timeout=None):
            order.append("communicate")
            raise subprocess.TimeoutExpired(
                "pytest", timeout,
                output=b"tests/test_a.py .\ntests/test_hangs_here.py ",
                stderr=b"",
            )

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    class _FakeContainer:
        def spawn(self, argv, **kwargs):
            return _StuckProc()

        def reap(self):
            order.append("reap")
            return ""

        def close(self):
            order.append("close")

    monkeypatch.setattr(process_containment, "ProcessContainer", _FakeContainer)
    monkeypatch.setattr(pr, "_terminate_preflight_tree", lambda proc, temp_root: None)

    returncode, output, reap_error = pr._execute_pytest_pass(
        sys.executable, tmp_path, tmp_path, ["tests/"], 0.01
    )

    assert returncode is None, "a timed-out pass must report no exit code"
    assert reap_error == ""
    assert "test_hangs_here" in output, (
        "both collections timed out and the excerpt was thrown away with them"
    )
    assert order[:3] == ["communicate", "reap", "communicate"], (
        f"the container was not reaped before the retry that it unblocks: {order}"
    )

@pytest.mark.parametrize("max_output", [1, 60, 200, 8000])
def test_timeout_excerpt_stays_inside_the_budget_and_keeps_the_tail(max_output):
    """The excerpt shares the caller's 8000-char limit with the message, and the
    caller re-truncates from the TAIL — so it must be bounded here, and it must
    keep the END of the output (progress output stops at the test that never
    finished), not the beginning.

    The bound is UNCONDITIONAL, matching `_diagnosis`: when the message alone
    already fills the budget the message is cut rather than returned whole. An
    earlier revision exempted that case, which made the documented invariant one
    with an exception — and `max_output` is a caller-declared limit, so the one
    thing it must never do is depend on which branch ran."""
    from ouroboros.preflight_runner import _with_timeout_excerpt

    message = "⚠️ PRE_PUSH_TEST_ERROR: pytest timed out after 30 seconds in the serial pass"
    output = ("noise line that is not the answer\n" * 400) + "tests/test_hangs_here.py "
    result = _with_timeout_excerpt(message, output, max_output)

    assert len(result) <= max_output, f"the excerpt overran the {max_output}-char budget"
    assert result.startswith(message) or result == message[:max_output], (
        "the excerpt displaced the message it explains"
    )
    if len(result) > len(message):
        assert result.rstrip().endswith("tests/test_hangs_here.py"), "kept the head instead of the tail"

def test_timeout_message_survives_an_empty_or_missing_excerpt():
    """A pass killed before it flushed anything has no excerpt, and appending an
    empty one would leave a dangling header promising output that never comes."""
    from ouroboros.preflight_runner import _with_timeout_excerpt

    message = "⚠️ PRE_PUSH_TEST_ERROR: pytest timed out"
    assert _with_timeout_excerpt(message, "", 8000) == message
    assert _with_timeout_excerpt(message, "   \n  ", 8000) == message

@requires_preflight_plugins
def test_hermetic_pytest_applies_candidate_diff_and_scrubs_live_env(tmp_path, monkeypatch, two_pass_env):
    """BOTH passes must see the candidate diff and the scrubbed env — a probe in
    only one lane would leave the other lane's wiring unproven."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    # 20-space indent: `_make_repo` dedents the 16-space template around it, so
    # these land one level in, inside the probe function body.
    assertions = "\n".join(
        " " * 20 + line for line in [
            'assert value.FLAG is True',
            'assert extra_value.FLAG is True',
            'assert "OUROBOROS_MANAGED_BY_LAUNCHER" not in os.environ',
            'assert "OUROBOROS_SAFETY_MODE" not in os.environ',
            'assert "OUROBOROS_TASK_REVIEW_MODE" not in os.environ',
            'assert "OUROBOROS_FAKE_API_KEY" not in os.environ',
            'assert "ouroboros-preflight-" in os.environ["OUROBOROS_DATA_DIR"]',
            'assert os.environ["OUROBOROS_SETTINGS_PATH"].startswith(os.environ["OUROBOROS_DATA_DIR"])',
            'assert "ouroboros-preflight-" in os.environ["OUROBOROS_REPO_DIR"]',
        ]
    )
    repo = _make_repo(
        tmp_path,
        {
            "value.py": "FLAG = False\n",
            "tests/test_parallel_lane.py": f"""
                import os
                import extra_value
                import value


                def test_candidate_diff_and_env_are_hermetic():
{assertions}
            """,
            "tests/test_serial_lane.py": f"""
                import os

                import pytest

                import extra_value
                import value


                @pytest.mark.serial
                def test_candidate_diff_and_env_are_hermetic_in_serial_pass():
{assertions}
            """,
        },
    )
    # Candidate (uncommitted) changes: a tracked edit plus an untracked new file.
    (repo / "value.py").write_text("FLAG = True\n", encoding="utf-8")
    (repo / "extra_value.py").write_text("FLAG = True\n", encoding="utf-8")

    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "light")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_FAKE_API_KEY", "must-not-reach-tests")
    result = run_hermetic_pytest(repo, timeout=120)

    assert result is None, result

@requires_preflight_plugins
def test_both_passes_execute_and_partition(tmp_path, two_pass_env):
    """One worktree, one env, two passes: the unmarked probe must run under an
    xdist worker and the serial probe must NOT — the lane partition IS the gate."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    marker_parallel = tmp_path / "ran_parallel.txt"
    marker_serial = tmp_path / "ran_serial.txt"
    repo = _make_repo(
        tmp_path,
        {
            "tests/test_parallel_probe.py": f"""
                import os


                def test_runs_in_the_parallel_pass():
                    assert os.environ.get("PYTEST_XDIST_WORKER"), "expected an xdist worker"
                    open(r'{marker_parallel}', "w").write(os.environ["PYTEST_XDIST_WORKER"])
            """,
            "tests/test_serial_probe.py": f"""
                import os

                import pytest


                @pytest.mark.serial
                def test_runs_in_the_serial_pass():
                    assert "PYTEST_XDIST_WORKER" not in os.environ, "serial test ran under xdist"
                    open(r'{marker_serial}', "w").write("serial")
            """,
        },
    )

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is None, result
    assert marker_parallel.exists(), "the parallel pass never ran its probe"
    assert marker_serial.exists(), "the serial pass never ran its probe"

@requires_preflight_plugins
def test_the_parallel_pass_really_starts_more_than_one_worker(tmp_path, two_pass_env):
    """A "parallel" pass on ONE worker exercises no concurrency, yet the argv
    still says `-n`, `PreflightPass.parallel` stays True and the green return is
    accepted as proof. `-n auto` resolves through PYTEST_XDIST_AUTO_NUM_WORKERS,
    which the operator environment can carry — `two_pass_env` pins it to "1",
    exactly the inherited downgrade the scrub must defeat — so this asserts the
    property behaviourally, from inside the candidate suite."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    workers_dir = tmp_path / "observed_workers"
    workers_dir.mkdir()
    # `--dist loadscope` distributes by MODULE, so one file is one work unit.
    # xdist's initial round-robin hands the first unit to each node, so four
    # units against the two-worker floor makes both nodes report.
    files = {}
    for index in range(4):
        files[f"tests/test_scope_{index}.py"] = f"""
            import os
            import pathlib


            def test_records_its_worker():
                # The operator's downgrade never arrived: the count is the one
                # the gate chose, and it is at least two.
                assert int(os.environ["PYTEST_XDIST_AUTO_NUM_WORKERS"]) >= 2
                worker = os.environ["PYTEST_XDIST_WORKER"]
                pathlib.Path(r'{workers_dir}', worker).write_text(worker)
        """
    repo = _make_repo(tmp_path, files)

    assert run_hermetic_pytest(repo, timeout=180) is None

    observed = sorted(path.name for path in workers_dir.iterdir())
    assert len(observed) >= 2, f"the parallel lane ran on a single worker: {observed}"

@requires_preflight_plugins
def test_a_candidate_cannot_switch_the_parallel_plugins_off(tmp_path, two_pass_env):
    """Verifying the interpreter proves the plugins are INSTALLED. It says nothing
    about whether the candidate's own pytest configuration lets them LOAD, and ini
    `addopts` are PREPENDED to the gate's argv, so `-p no:xdist -p no:timeout` in
    the candidate's `pytest.ini` disarms both before the gate's flags are read.

    That is why the parallel pass appends `-p xdist -p timeout`: `consider_preparse`
    walks `-p` entries in order, so the later unblock wins over the earlier block.
    Pinned behaviourally — the candidate says no, and the lane still fans out over
    at least two real workers."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    workers_dir = tmp_path / "hostile_workers"
    workers_dir.mkdir()
    files = {
        "pytest.ini": (
            "[pytest]\n"
            "addopts = -p no:xdist -p no:timeout\n"
            "markers =\n"
            "    serial: real-process/port/global-state test; runs in the serial pass\n"
        ),
    }
    # `--dist loadscope` distributes by MODULE, so four files are four work units
    # and xdist's initial round-robin reaches both nodes.
    for index in range(4):
        files[f"tests/test_scope_{index}.py"] = f"""
            import os
            import pathlib


            def test_records_its_worker():
                worker = os.environ["PYTEST_XDIST_WORKER"]
                pathlib.Path(r'{workers_dir}', worker).write_text(worker)
        """
    repo = _make_repo(tmp_path, files)

    assert run_hermetic_pytest(repo, timeout=180) is None

    observed = sorted(path.name for path in workers_dir.iterdir())
    assert len(observed) >= 2, (
        f"the candidate switched xdist off and the gate accepted it: {observed}"
    )

@requires_preflight_plugins
def test_a_candidate_faking_the_parallel_flags_cannot_earn_a_green_pass(tmp_path, two_pass_env):
    """The full attack, end to end. `pytest.ini` blocks both plugins AND a conftest
    declares the gate's own flags with `pytest_addoption` and ignores them, so
    nothing rejects `-n`: the lane is labelled parallel, runs strictly serially and
    exits 0. Green, on a pass that never ran two things at once.

    The invariant is that this shape cannot be BOTH green and serial. Forcing the
    plugins on makes the conftest's duplicate option definitions collide with the
    real ones, which is a usage error and a block; without the collision the lane
    genuinely fans out. Either outcome is fail-closed — a silent pass is not."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    workers_dir = tmp_path / "faked_workers"
    workers_dir.mkdir()
    files = {
        "pytest.ini": (
            "[pytest]\n"
            "addopts = -p no:xdist -p no:timeout\n"
            "markers =\n"
            "    serial: real-process/port/global-state test; runs in the serial pass\n"
        ),
        "conftest.py": """
            def pytest_addoption(parser):
                # Swallow the gate's parallel flags so a plugin-less pytest accepts
                # them: "pytest did not reject -n" is not evidence of parallelism.
                parser.addoption("-n", "--numprocesses", action="store", default=None)
                parser.addoption("--dist", action="store", default=None)
                parser.addoption("--timeout", action="store", default=None)
                parser.addoption("--timeout-method", action="store", default=None)
                parser.addoption("--max-worker-restart", action="store", default=None)
        """,
    }
    for index in range(4):
        files[f"tests/test_scope_{index}.py"] = f"""
            import os
            import pathlib


            def test_records_its_worker():
                worker = os.environ.get("PYTEST_XDIST_WORKER", "none")
                pathlib.Path(r'{workers_dir}', worker).write_text(worker)
        """
    repo = _make_repo(tmp_path, files)

    result = run_hermetic_pytest(repo, timeout=180)

    observed = sorted(path.name for path in workers_dir.iterdir())
    assert result is not None or len(observed) >= 2, (
        f"a lane that never ran in parallel returned green; workers observed: {observed}"
    )

@requires_preflight_plugins
@pytest.mark.skipif(os.name == "nt", reason="POSIX marker-membership containment")
def test_a_green_pass_cannot_leak_a_child_into_the_next_pass(tmp_path, two_pass_env):
    """Containment must be UNCONDITIONAL, not a timeout/crash path.

    `communicate()` returning only proves the pytest CONTROLLER exited. A test
    that spawned a child and did not wait for it leaves that child alive, and
    after the controller dies nothing can find it: the `pgrep -P` parent->child
    walk is gone with the ppid links, and the temp-root command-line sweep misses
    an argv that names no sweepable path. Such a child ran on into pass 2 — the
    very cross-pass contamination the inter-pass sweep exists to prevent — and
    then past teardown onto the machine.

    The child here calls `setsid()` (`start_new_session=True`), which is the
    HARDEST shape: it leaves the controller's process group, so the recorded pgid
    no longer names it, and it is the shape a daemonising child naturally takes.
    Only the container's environment membership token — which the kernel copied
    into the child at fork and which `setsid()`, orphaning and closed stdio all
    leave untouched — can still name it once the controller is gone.

    The probe returns IMMEDIATELY after spawning. An earlier revision recorded
    descendants from a 0.5s background poll and this test slept for two seconds
    so a sample would land while the ppid link still existed; that sleep was the
    test accommodating the defect, since a green pass that spawns and returns
    fast is precisely the leak. Membership is resolved from live kernel state at
    reap time, so no sampling has to happen at all.

    Pinned end to end: pass 1 is green, and pass 2 (a different pytest process
    entirely) observes the pid already dead."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    marker = tmp_path / "escapee.pid"
    repo = _make_repo(
        tmp_path,
        {
            # Stdio to DEVNULL so the child does NOT hold the inherited pipe open
            # — otherwise `communicate` blocks and this becomes the timeout case
            # that was already covered. The argv is deliberately path-free, so the
            # temp-root sweep cannot see it either. `start_new_session=True` puts
            # it in its OWN session and process group, so the group handle the
            # container recorded at spawn does not cover it.
            "tests/test_leaks_a_child.py": f"""
                import pathlib
                import subprocess
                import sys


                def test_spawns_a_child_and_passes():
                    child = subprocess.Popen(
                        [sys.executable, "-c", "import time; time.sleep(180)"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    pathlib.Path(r'{marker}').write_text(str(child.pid))
                    # No sleep: the test returns at once, so the child is born,
                    # detached and orphaned with nothing observing it. That is the
                    # shape the containment must survive.
            """,
            "tests/test_checks_the_child_is_gone.py": f"""
                import os
                import pathlib
                import time

                import pytest


                @pytest.mark.serial
                def test_the_parallel_pass_left_nothing_running():
                    pid = int(pathlib.Path(r'{marker}').read_text().strip())
                    deadline = time.time() + 15
                    while time.time() < deadline:
                        try:
                            os.kill(pid, 0)
                        except OSError:
                            return
                        time.sleep(0.2)
                    raise AssertionError(
                        "pid %d from the parallel pass is still alive in the serial pass" % pid
                    )
            """,
        },
    )

    try:
        result = run_hermetic_pytest(repo, timeout=180)
        assert result is None, result
        assert marker.exists(), "the parallel probe never spawned its child"
    finally:
        # A containment regression must not leak a 180s sleeper into the suite.
        if marker.exists():
            leaked = int(marker.read_text().strip())
            if pid_is_alive(leaked):
                force_kill_pid(leaked)

@requires_preflight_plugins
def test_worker_crash_is_hard_block(tmp_path, two_pass_env):
    """A dead xdist worker is a HARD BLOCK with mark-it-serial remediation, never
    an ordinary failure and never a retryable flake. Fail-fast: no pass 2."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    marker_serial = tmp_path / "ran_serial.txt"
    repo = _make_repo(
        tmp_path,
        {
            "tests/test_crash_probe.py": """
                import os


                def test_kills_its_worker():
                    os._exit(1)
            """,
            "tests/test_serial_probe.py": f"""
                import pytest


                @pytest.mark.serial
                def test_should_never_run():
                    open(r'{marker_serial}', "w").write("serial")
            """,
        },
    )

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is not None, "a crashed worker must block the commit"
    assert "PARALLEL_WORKER_CRASH" in result, result
    assert "@pytest.mark.serial" in result, result
    assert "never a flake/retry" in result, result
    assert not marker_serial.exists(), "fail-fast broken: the serial pass ran after a red pass 1"

@requires_preflight_plugins
def test_empty_serial_lane_is_green(tmp_path, two_pass_env):
    """Exit 5 is green PER PASS — a candidate repo with zero serial tests must
    not be false-blocked by an empty serial lane."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(
        tmp_path,
        {
            "tests/test_plain.py": """
                def test_ok():
                    assert True
            """,
        },
    )

    assert run_hermetic_pytest(repo, timeout=120) is None

@requires_preflight_plugins
def test_both_lanes_empty_blocks(tmp_path, two_pass_env):
    """...but a `tests/` directory that yields NO runnable test in ANY pass keeps
    blocking, preserving the empty-suite invariant."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(
        tmp_path,
        {
            "tests/helpers.py": """
                def not_a_test():
                    return True
            """,
        },
    )

    result = run_hermetic_pytest(repo, timeout=120)
    assert result is not None
    assert "no tests were collected" in result, result

@requires_preflight_plugins
def test_pass2_timeout_names_serial_pass(tmp_path, two_pass_env):
    """The 900s budget is TOTAL; pass 2 gets the remainder and its timeout must
    name the pass so a hung serial test is not mistaken for a hung parallel one."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(
        tmp_path,
        {
            "tests/test_fast.py": """
                def test_ok():
                    assert True
            """,
            "tests/test_slow_serial.py": """
                import time

                import pytest


                @pytest.mark.serial
                def test_hangs():
                    time.sleep(300)
            """,
        },
    )

    result = run_hermetic_pytest(repo, timeout=30)

    assert result is not None
    assert "timed out" in result, result
    assert "serial pass" in result, result
    assert "total budget 30 seconds" in result, result

def test_hermetic_pytest_timeout_invokes_full_tree_reaper():
    """The timeout path must delegate to the full-tree reaper (not a bare killpg),
    and that reaper must use the recursive PID-tree kill, escaped process-group
    kill, and the temp-root sweep so detached/reparented children cannot survive."""
    from ouroboros import preflight_runner

    pass_src = inspect.getsource(preflight_runner._execute_pytest_pass)
    assert "_terminate_preflight_tree" in pass_src

    # The process is spawned INSIDE the container, never `Popen`'d and adopted
    # afterwards: on Windows job membership only takes effect at assignment, so a
    # descendant started in that window is outside the job and survives
    # terminate/close — the exact leak the container exists to close.
    assert "container.spawn(" in pass_src
    assert "subprocess.Popen(" not in pass_src, (
        "spawning outside the container reopens the Windows job-assignment race"
    )

    # The container is reaped UNCONDITIONALLY — including after a GREEN pass,
    # which is exactly the case the `proc.poll() is None` guard skips and the case
    # `test_a_green_pass_cannot_leak_a_child_into_the_next_pass` covers. Pinned in
    # source too, because that behavioural test is POSIX-only and Windows relies
    # on the same call reaching the Job Object.
    #
    # It happens BEFORE the return, not only in `finally`: a `finally` block
    # cannot alter an already-computed return tuple, so a reap that runs only
    # there can report a containment FAILURE that no caller can ever see — which
    # is the fail-open the container exists to close. `finally` still carries a
    # reap (for the raising path, where there is no verdict to carry it) and the
    # handle release.
    returning, _, teardown = pass_src.partition("finally:")
    assert "reap_error = container.reap()" in returning, (
        "the reap result cannot reach the caller, so containment fails open"
    )
    assert "return returncode, output, reap_error" in returning
    assert "container.reap()" in teardown
    assert "container.close()" in teardown

    # The inter-pass temp-root sweep is pinned BEHAVIOURALLY by
    # `test_temp_root_is_swept_between_passes_not_only_at_teardown`, not here: a
    # bare `"kill_processes_referencing" in run_hermetic_pytest source` check
    # cannot fail, because the teardown `finally` block contains that same call
    # and predates the two-pass split.
    reaper_src = inspect.getsource(preflight_runner._terminate_preflight_tree)
    assert "kill_process_tree" in reaper_src
    assert "kill_pid_tree" in reaper_src
    assert "kill_process_group_id" in reaper_src
    assert "kill_processes_referencing" in reaper_src
    # Platform-specific process discovery stays behind platform_layer helpers.
    assert "collect_descendant_pids" in reaper_src

def test_resolve_preflight_timeout_env_override(monkeypatch):
    """`OUROBOROS_PREFLIGHT_TIMEOUT_SEC` overrides the TOTAL two-pass budget."""
    from ouroboros.preflight_runner import _resolve_preflight_timeout

    monkeypatch.delenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", raising=False)
    assert _resolve_preflight_timeout(300) == 300
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "450")
    assert _resolve_preflight_timeout(300) == 450
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "not-an-int")
    assert _resolve_preflight_timeout(300) == 300
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "0")
    assert _resolve_preflight_timeout(300) == 300

def test_hermetic_pytest_prefers_agent_python_env():
    from ouroboros import preflight_runner

    runner_src = inspect.getsource(preflight_runner.run_hermetic_pytest)
    assert 'os.environ.get("OUROBOROS_AGENT_PYTHON") or sys.executable' in runner_src
    # ...and every pass is actually spawned with that resolved interpreter.
    pass_src = inspect.getsource(preflight_runner._execute_pytest_pass)
    assert '[agent_python, "-m", "pytest", *args]' in pass_src

@requires_preflight_plugins
@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group reaping behaviour")
def test_hermetic_pytest_timeout_reaps_detached_session_child(tmp_path, two_pass_env):
    """A child that escapes pytest's process group (its own session) must still be
    reaped on timeout — the orphan class the QA hit (96% CPU survivor)."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    marker = tmp_path / "child.pid"
    # NOTE: this fixture is generated user-style code that runs inside the
    # hermetic worktree (where `ouroboros` is not on sys.path), so it must stay
    # stdlib-only. start_new_session=True simulates an arbitrary skill test
    # spawning a detached child in its own session — exactly the orphan class the
    # reaper must catch. The test HARNESS itself uses platform_layer helpers.
    # Unmarked, so it hangs inside the PARALLEL pass (one xdist worker deep).
    repo = _make_repo(
        tmp_path,
        {
            "tests/test_hang.py": f"""
                import sys, subprocess, time

                def test_spawns_detached_child_and_hangs():
                    # Detached child in its OWN session escapes the pytest group's killpg.
                    subprocess.Popen(
                        [sys.executable, "-c",
                         "import os,time;open(r'{marker}','w').write(str(os.getpid()));time.sleep(180)"],
                        start_new_session=True,
                    )
                    time.sleep(180)
            """,
        },
    )

    # 10s (was 5s serial): the parallel pass pays xdist controller+worker startup
    # before the probe body runs at all.
    result = run_hermetic_pytest(repo, timeout=10)
    assert result is not None and "timed out" in result

    assert marker.exists(), "detached child never recorded its pid"
    child_pid = int(marker.read_text().strip())
    deadline = time.time() + 10
    alive = True
    while time.time() < deadline:
        if not pid_is_alive(child_pid):
            alive = False
            break
        time.sleep(0.2)
    if alive:  # cleanup so a reaping regression does not leak a 180s sleeper
        force_kill_pid(child_pid)
    assert not alive, f"detached child {child_pid} survived preflight timeout reaping"
