"""Test runners spend a shared physical budget and preserve crash evidence."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

from ouroboros import config, preflight_runner, test_parallelism as budget
from ouroboros.process_containment import ProcessContainer, pid_is_zombie
from ouroboros import platform_layer as pl
from tests import conftest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_native_memory_reports_reclaimable_bytes_without_double_counting(monkeypatch, platform):
    monkeypatch.setattr(budget.sys, "platform", platform)
    if platform == "darwin":
        outputs = iter(["65536\n", 'Mach Virtual Memory Statistics: (page size of 4096 bytes)\n'
                        'Pages free: 2.\nPages inactive: 3.\nPages speculative: 1.\nPages purgeable: 2.\n',
                        "1\n"])  # kern.memorystatus_vm_pressure_level: normal
        monkeypatch.setattr(budget, "_native_output", lambda argv: next(outputs))
    elif platform == "linux":
        monkeypatch.setattr(budget.Path, "read_text", lambda self, **kwargs: "MemTotal: 64 kB\nMemAvailable: 24 kB\n")
        monkeypatch.setattr(budget.Path, "is_file", lambda self: False)  # no cgroup limit on this host
    else:
        monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(total=65536, available=24576)))
    assert budget.physical_memory() == budget.MemorySnapshot(65536, 24576)


@pytest.mark.parametrize("level, pressure", [("1\n", False), ("2\n", False), ("4\n", True), (None, False)])
def test_macos_kernel_pressure_counts_although_inactive_pages_look_free(monkeypatch, level, pressure):
    """XNU keeps refilling the inactive queue with anonymous pages it then compresses, so free +
    inactive + speculative barely moves under pressure: the kernel's critical level is read too.
    Warn is routine on a busy small Mac running a healthy battery. An unreadable level leaves the
    free-memory reading alone."""
    vm = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 2.\nPages inactive: 3.\nPages speculative: 1.\n"

    def native(argv):
        if argv[-1] == "kern.memorystatus_vm_pressure_level":
            if level is None:
                raise subprocess.CalledProcessError(1, argv)
            return level
        return "65536\n" if argv[-1] == "hw.memsize" else vm

    monkeypatch.setattr(budget.sys, "platform", "darwin")
    monkeypatch.setattr(budget, "_native_output", native)
    assert budget.physical_memory() == budget.MemorySnapshot(65536, 24576, pressure)


def test_native_probe_failure_admits_one_worker(monkeypatch):
    monkeypatch.setattr(budget.sys, "platform", "darwin")
    monkeypatch.setattr(budget, "_native_output", lambda argv: "unreadable")
    assert budget.physical_memory() is None
    assert budget.worker_budget(16, env={budget.WORKERS_ENV: "24"}) == 1


@pytest.mark.parametrize("requested, inherited, expected", [(16, None, 3), (2, None, 2), (16, 1, 1), (16, 2, 2)])
def test_physical_headroom_and_parent_slots_cap_the_same_budget(monkeypatch, requested, inherited, expected):
    peak = config.TEST_WORKER_PEAK_RSS_BYTES
    monkeypatch.setattr(budget, "physical_memory", lambda: budget.MemorySnapshot(10 * peak, 4 * peak))
    env = {} if inherited is None else {budget.NESTED_WORKERS_ENV: str(inherited)}
    assert budget.worker_budget(requested, env=env) == expected


@pytest.mark.parametrize("capacity", [2, 3])
def test_the_gate_pins_the_worker_count_it_verified(tmp_path, monkeypatch, capacity):
    """The requested three slots still obey measured memory, not a hostile pytest override."""
    total = 20 * config.TEST_WORKER_PEAK_RSS_BYTES
    memory = budget.MemorySnapshot(total, int(total * config.TEST_MEMORY_RESERVE_FRACTION)
                                   + capacity * config.TEST_WORKER_PEAK_RSS_BYTES)
    monkeypatch.setattr(budget, "physical_memory", lambda: memory)
    monkeypatch.setenv(budget.WORKERS_ENV, "3")
    monkeypatch.setenv(budget.NESTED_WORKERS_ENV, "3")
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "1")
    env = preflight_runner._preflight_env(tmp_path / "root", tmp_path / "root" / "repo")
    assert env["PYTEST_XDIST_AUTO_NUM_WORKERS"] == str(capacity)
    assert budget.WORKERS_ENV not in env  # the candidate cannot re-lower the admission seam


def test_preflight_keeps_the_admission_it_measured_before_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_runner, "_preflight_worker_count", lambda: 2)
    preflight_runner._install_worker_probe(tmp_path)
    monkeypatch.setattr(preflight_runner, "_preflight_worker_count", lambda: 16)
    env = preflight_runner._preflight_env(tmp_path, tmp_path / "repo")
    assert env["PYTEST_XDIST_AUTO_NUM_WORKERS"] == "2"
    assert env[budget.NESTED_WORKERS_ENV] == "2"


def test_local_node_lane_uses_the_same_cpu_and_memory_budget(tmp_path, monkeypatch):
    from ouroboros import preflight_node

    spec = importlib.util.spec_from_file_location("budget_test_runner", ROOT / "scripts/run_tests.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    peak = config.TEST_WORKER_PEAK_RSS_BYTES
    monkeypatch.setattr(budget, "physical_memory", lambda: budget.MemorySnapshot(10 * peak, 4 * peak))
    monkeypatch.setenv(budget.WORKERS_ENV, "16")
    monkeypatch.delenv(budget.NESTED_WORKERS_ENV)
    monkeypatch.setattr(preflight_node, "candidate_node_tests", lambda repo: ["tests/a.test.js", "tests/b.test.js"])
    monkeypatch.setattr(preflight_node, "resolve_node", lambda: "/fake/node")
    monkeypatch.setattr(preflight_node, "probe_node_version", lambda node: "24.0.0")
    calls = []
    monkeypatch.setattr(runner, "_run", lambda label, argv, cwd: calls.append((argv, cwd)) or 0)
    assert runner._node_lane() == 0
    assert calls == [(["/fake/node", "--test", "--test-concurrency=3", "tests/a.test.js", "tests/b.test.js"],
                      runner.REPO / "web")]


@pytest.mark.parametrize("argv, slots, expected", [
    (["node", "--test", "one.js"], None, ["node", "--test", "--test-concurrency=1", "one.js"]),
    (["node", "--test", "--test-concurrency=8", "one.js"], "2",
     ["node", "--test", "--test-concurrency=2", "one.js"]),
    (["python", "-m", "pytest", "-n", "auto"], None, ["python", "-m", "pytest", "-n", "1"]),
    (["python", "-m", "pytest", "-n2"], "2", ["python", "-m", "pytest", "-n2"]),
    (["python", "-m", "pytest", "--numprocesses=0"], None, ["python", "-m", "pytest", "--numprocesses=0"]),
    (["python", "-m", "pytest", "--numprocesses=bad"], None, ["python", "-m", "pytest", "--numprocesses=bad"]),
    (["python", "-m", "pytest", "-nonsense"], None, ["python", "-m", "pytest", "-nonsense"]),
    (["node", "--test", "--test-concurrency=bad"], None, ["node", "--test", "--test-concurrency=bad"]),
    (["node", "--test", "--test-concurrency"], None, ["node", "--test", "--test-concurrency"]),
])
def test_shared_child_spawn_keeps_containment_and_bounds_nested_runners(monkeypatch, argv, slots, expected):
    seen = {}
    token = "OURO_PROC_CONTAINER_1234567890ABCDEF"  # Windows upper-cases os.environ keys
    monkeypatch.setenv(token, "1")
    monkeypatch.setattr(conftest, "_ORIGINAL_POPEN_INIT", lambda self, args, **kwargs: seen.update(args=args, **kwargs))
    env = {} if slots is None else {budget.NESTED_WORKERS_ENV: slots}
    subprocess.Popen(argv, env=env)
    assert seen["args"] == expected
    assert seen["env"][token] == "1"
    assert seen["env"][budget.NESTED_WORKERS_ENV] == (slots or "1")
    assert seen["env"]["PYTEST_XDIST_AUTO_NUM_WORKERS"] == (slots or "1")


def _run_pytest(tmp_path, files, *, workers, extra=(), owned=True):
    suite = tmp_path / "suite"
    suite.mkdir()
    for name, body in files.items():
        (suite / name).write_text(textwrap.dedent(body), encoding="utf-8")
    evidence, log = tmp_path / "evidence", tmp_path / "run.log"
    args = ["-p", "tests.conftest", "-o", "addopts=", "--rootdir", str(suite), "--confcutdir", str(suite),
            "--ci-evidence-dir", str(evidence), "-n", "2", "--dist", "loadscope", "--max-worker-restart=0",
            "-q", *extra, str(suite)]
    entry = tmp_path / "run.py"
    entry.write_text(f"import sys\nsys.path.insert(0, {str(ROOT)!r})\nimport pytest\n"
                     f"raise SystemExit(pytest.main({args!r}))\n", encoding="utf-8")
    started = time.monotonic()
    if not owned:  # A bare run: the caller removed every inherited membership token.
        # No launcher root either: its own evidence in tmp_path would otherwise retain the outer one.
        env = {key: value for key, value in os.environ.items() if key != "OUROBOROS_TEST_TEMP_ROOT"}
        with log.open("w", encoding="utf-8") as output:
            # Its own session, as every owned child: the kill below then reaches this run, never ours (#2353).
            proc = subprocess.Popen([sys.executable, str(entry)], cwd=ROOT,
                                    env={**env, budget.NESTED_WORKERS_ENV: str(workers)},
                                    stdout=output, stderr=subprocess.STDOUT, **pl.subprocess_new_group_kwargs())
        try:
            code = proc.wait(timeout=60)
        finally:
            pl.kill_process_tree(proc)
            proc.wait(timeout=10)
        return code, evidence, log.read_text(encoding="utf-8"), time.monotonic() - started
    container = ProcessContainer(lifeline=True, overload=True)
    command = [sys.executable, "-I", "-S", str(ROOT / "scripts/safe_test.py"), "--temp-parent", str(tmp_path),
               "--", sys.executable, str(entry)]
    with log.open("w", encoding="utf-8") as output:
        proc = container.spawn(command, cwd=ROOT, env={**os.environ, budget.NESTED_WORKERS_ENV: str(workers)},
                               stdout=output, stderr=subprocess.STDOUT)
    try:
        code = proc.wait(timeout=60)
    finally:
        pl.kill_process_tree(proc)
        reap_error = container.reap()
        container.close()
        proc.wait(timeout=10)
    assert not reap_error, reap_error
    return code, evidence, log.read_text(encoding="utf-8"), time.monotonic() - started


def test_a_cgroup_limit_bounds_physical_memory_and_credits_reclaimable_cache(monkeypatch):
    gib = 1024 ** 3
    files = {"/proc/meminfo": "MemTotal: 33554432 kB\nMemAvailable: 16777216 kB\n",  # a 32 GiB host, 16 free
             "/sys/fs/cgroup/memory.max": f"{4 * gib}\n",  # a 4 GiB container
             "/sys/fs/cgroup/memory.current": f"{3 * gib}\n",  # 3 GiB charged, 1 GiB of it file cache
             "/sys/fs/cgroup/memory.stat": f"anon {2 * gib}\ninactive_file {gib}\n"}
    real_read, real_is_file = Path.read_text, Path.is_file

    def faked(path):  # POSIX spelling: a Windows host renders these paths with backslashes
        return path.as_posix().startswith(("/proc/", "/sys/fs/cgroup/"))

    def read_text(self, *args, **kwargs):
        if not faked(self):
            return real_read(self, *args, **kwargs)
        if self.as_posix() not in files:  # absent here, never this host's real cgroup file
            raise FileNotFoundError(self.as_posix())
        return files[self.as_posix()]

    monkeypatch.setattr(budget, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "is_file", lambda self: self.as_posix() in files if faked(self) else real_is_file(self))
    memory = budget.physical_memory()
    assert (memory.total_bytes, memory.available_bytes) == (4 * gib, 2 * gib)  # the cap, minus anon only
    del files["/sys/fs/cgroup/memory.stat"]  # without the stat file the cache is not credited
    assert budget.physical_memory().available_bytes == gib


@pytest.mark.parametrize("slots", [1, 2])
def test_generated_pytest_entry_obeys_nested_allocation(tmp_path, slots):
    workers = tmp_path / "workers"
    workers.mkdir()
    code, _, output, _ = _run_pytest(tmp_path, {
        "conftest.py": f"""
            from pathlib import Path
            def pytest_configure(config):
                if hasattr(config, 'workerinput'):
                    worker = config.workerinput['workerid']
                    (Path({str(workers)!r}) / worker).write_text(worker, encoding='utf-8')
        """,
        "test_case.py": "def test_case(): assert True\n",
    }, workers=slots)
    assert code == 0, output
    assert len(list(workers.iterdir())) == slots, output


@pytest.mark.parametrize("fault", ["exit", "thread_timeout"])
def test_worker_death_ends_pass_with_crash_and_unfinished_evidence(tmp_path, fault):
    child_identity, running, finished = tmp_path / "child.json", tmp_path / "running", tmp_path / "finished"
    child_code = ("import os,time,json; from pathlib import Path; "
                  "from ouroboros.platform_layer import process_start_time; "
                  f"Path({str(child_identity)!r}).write_text(json.dumps([os.getpid(),process_start_time(os.getpid())]),encoding='utf-8'); "
                  "time.sleep(40)")
    crash = "os._exit(7)" if fault == "exit" else "time.sleep(40)"
    code, evidence, output, _ = _run_pytest(tmp_path, {
        "test_0_crash.py": f"""
            import os, subprocess, sys, time
            from pathlib import Path
            def test_dies():
                subprocess.Popen([sys.executable, '-c', {child_code!r}])
                deadline = time.monotonic() + 10
                while not Path({str(running)!r}).exists() or not Path({str(child_identity)!r}).exists():
                    assert time.monotonic() < deadline
                    time.sleep(.01)
                {crash}
        """,
        "test_1_running.py": f"""
            import pytest, time
            from pathlib import Path
            @pytest.mark.timeout(0)
            def test_running():
                Path({str(running)!r}).write_text('started', encoding='utf-8')
                time.sleep(40)
                Path({str(finished)!r}).write_text('finished', encoding='utf-8')
        """,
        "test_2_pending.py": "def test_pending(): raise AssertionError('must not schedule after worker death')\n",
    }, workers=2, extra=("--timeout=3", "--timeout-method=thread") if fault == "thread_timeout" else ())
    assert code != 0 and not finished.exists(), output  # the pass ended without waiting for the sleeper
    data = json.loads((evidence / "results.json").read_text(encoding="utf-8"))
    crash_reports = [row for row in data["reports"] if row["phase"] == "crash"]
    assert any(row["nodeid"] == "test_0_crash.py::test_dies" for row in crash_reports), output
    # Only the timeout callback's own record names a cause; an os._exit leaves the crash unexplained.
    assert [row.get("timeout_callback_seconds") for row in crash_reports] == [
        3.0 if fault == "thread_timeout" else None], output
    assert not any(row["nodeid"] == "test_2_pending.py::test_pending" for row in data["reports"]), output
    assert data["session_exit_code"] != 0
    assert data["worker_crash"]["collection_available"] is True
    assert data["worker_crash"]["unfinished"] == [
        "test_1_running.py::test_running", "test_2_pending.py::test_pending"]
    pid, born = json.loads(child_identity.read_text(encoding="utf-8"))
    # Gone or an unreaped corpse. Windows keeps an exited process (and its birth) readable
    # while a handle is open and ends Job members asynchronously, so its exit state decides.
    deadline = time.monotonic() + 15
    while pl.process_start_time(pid) == born and not pl.pid_provably_gone(pid) and not pid_is_zombie(pid):
        assert time.monotonic() < deadline, output
        time.sleep(0.05)


@pytest.mark.parametrize("name, change, seconds", [
    ("gw0", {}, 300.0),
    ("gw0", {"testrunuid": "an-earlier-run"}, None),
    ("gw0", {"worker": "gw1"}, None),
    ("gw1", {"worker": "gw1"}, None),  # another worker's own record stays with that worker
    ("gw0", {"nodeid_sha256": hashlib.sha256(b"t.py::test_other").hexdigest()}, None),
    ("gw0", {"timeout_seconds": "300"}, None),
    ("gw0", "{torn", None),
    ("gw0", None, None),
], ids=["exact", "stale-run", "names-another-worker", "another-workers-file", "another-test", "malformed",
        "torn", "absent"])
def test_a_crash_names_a_timeout_callback_only_from_its_own_workers_record(tmp_path, name, change, seconds):
    from tests.ci_evidence import _timeout_callback

    report = SimpleNamespace(nodeid="t.py::test_dies", node=SimpleNamespace(
        gateway=SimpleNamespace(id="gw0"), workerinput={"testrunuid": "this-run"}))
    record = tmp_path / f".timeout-{name}.json"
    if change is not None:
        exact = {"worker": "gw0", "testrunuid": "this-run", "timeout_seconds": 300.0,
                 "nodeid_sha256": hashlib.sha256(b"t.py::test_dies").hexdigest()}
        record.write_text(change if isinstance(change, str) else json.dumps({**exact, **change}), encoding="utf-8")
    assert _timeout_callback(tmp_path, report) == seconds
    # The dead worker's record is read once; another worker's is left for its own crash.
    assert record.exists() is (name != "gw0" and change is not None)


def test_unowned_worker_death_lets_running_tests_finish(tmp_path, monkeypatch):
    """With no outer owner to reap them, xdist's own shutdown lets running tests finish."""
    from ouroboros.process_containment import CONTAINMENT_ENV_PREFIX

    for key in [key for key in os.environ if key.startswith(CONTAINMENT_ENV_PREFIX)]:
        monkeypatch.delenv(key)
    finished = tmp_path / "running-finished"
    code, evidence, output, _ = _run_pytest(tmp_path, {
        "test_0_crash.py": """
            import os, time
            def test_dies():
                time.sleep(1)
                os._exit(7)
        """,
        "test_1_running.py": f"""
            import time
            from pathlib import Path
            def test_running():
                time.sleep(15)  # longer than xdist's ten-second teardown wait before it kills workers
                Path({str(finished)!r}).write_text('finished', encoding='utf-8')
        """,
    }, workers=2, owned=False)
    assert code == 1, output  # xdist's own crash handling: a failed run, not an interrupted one
    assert finished.exists(), output
    data = json.loads((evidence / "results.json").read_text(encoding="utf-8"))
    assert data["worker_crash"]["reason"].startswith("xdist worker")
    assert data["worker_crash"]["unfinished"] == []  # recorded at the end, after the survivor finished


@pytest.mark.parametrize("phase", ["collection", "idle"])
def test_worker_death_without_an_active_case_stops_other_workers(tmp_path, phase):
    finished, fast = tmp_path / "other-worker-finished", tmp_path / "fast-worker"
    hook = "pytest_collection" if phase == "collection" else "pytest_sessionfinish"
    code, evidence, output, _ = _run_pytest(tmp_path, {
        "conftest.py": f"""
            import os, time
            from pathlib import Path
            def {hook}(session):
                worker = getattr(session.config, 'workerinput', {{}}).get('workerid')
                # Idle: the worker that ran the fast module dies while the other still sleeps;
                # which worker loadscope hands that module to is not fixed.
                dies = (Path({str(fast)!r}).exists() and Path({str(fast)!r}).read_text(encoding='utf-8') == worker
                        if {phase!r} == 'idle' else worker == 'gw0')
                if dies:
                    os._exit(7)
                if worker == 'gw1' and {phase!r} == 'collection':
                    time.sleep(40)
                    Path({str(finished)!r}).write_text('finished', encoding='utf-8')
        """,
        "test_0_fast.py": f"""
            import os
            from pathlib import Path
            def test_fast():
                Path({str(fast)!r}).write_text(os.environ['PYTEST_XDIST_WORKER'], encoding='utf-8')
        """,
        "test_1_wait.py": f"""
            import time
            from pathlib import Path
            def test_wait():
                time.sleep(40)
                Path({str(finished)!r}).write_text('finished', encoding='utf-8')
        """,
    }, workers=2)
    assert code != 0 and not finished.exists(), output
    data = json.loads((evidence / "results.json").read_text(encoding="utf-8"))
    assert data["worker_crash"]["worker"] == (fast.read_text(encoding="utf-8") if phase == "idle" else "gw0")
    assert data["worker_crash"]["collection_available"] is (phase == "idle")
    if phase == "idle":
        assert data["worker_crash"]["unfinished"] == ["test_1_wait.py::test_wait"]
