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

from ouroboros import config, host_memory, preflight_runner, test_parallelism as budget
from ouroboros.process_containment import ProcessContainer
from ouroboros import platform_layer as pl
from tests import conftest
from ouroboros.process_containment import pid_is_zombie

ROOT = Path(__file__).resolve().parents[1]


def test_native_probe_failure_admits_one_worker(monkeypatch):
    monkeypatch.setattr(host_memory, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(host_memory, "_native_output", lambda argv: "unreadable")
    assert budget.physical_memory() is None
    assert budget.admission(16, env={budget.WORKERS_ENV: "24"}).workers == 1


@pytest.mark.parametrize("requested, inherited, host, expected", [
    (16, None, budget.HostRuns(), 3), (2, None, budget.HostRuns(), 2), (16, 1, budget.HostRuns(), 1),
    (16, 2, budget.HostRuns(), 2),
    # What other live runs hold comes off the top: leased slots, one per unleased root; floor one.
    (16, None, budget.HostRuns(slots=2, leases=((100, "run_tests.py overlapped"),)), 1),
    (16, None, budget.HostRuns(strangers=((700, "python3 -m pytest -n 8"),)), 2),
    (16, None, budget.HostRuns(slots=5), 1),
    (16, 2, None, 2),  # a child with a parent allocation reads no ledger: its parent accounted for it
])
def test_physical_headroom_parent_slots_and_other_runs_cap_the_same_budget(monkeypatch, requested, inherited,
                                                                           host, expected):
    peak = config.TEST_WORKER_PEAK_RSS_BYTES
    monkeypatch.setattr(budget, "physical_memory", lambda: host_memory.MemorySnapshot(10 * peak, 4 * peak))

    def census():
        assert host is not None, "a child with a parent allocation reads no ledger and runs no census"
        return host

    monkeypatch.setattr(budget, "other_runs", census)
    env = {} if inherited is None else {budget.NESTED_WORKERS_ENV: str(inherited)}
    assert budget.admission(requested, env=env) == budget.Admission(expected, host or budget.HostRuns())


def test_the_ledger_counts_live_leases_and_the_census_counts_unleased_roots(tmp_path, monkeypatch):
    """A live lease outside this lineage holds its slots; a dead, reborn or unreadable row counts nothing
    and the next writer unlinks it; every pytest/node root nobody leased counts one slot, while a nested
    run, a leased owner's tree and this process's own run (its ancestors included) never do."""
    ledger, me = tmp_path / "slots", os.getpid()
    monkeypatch.setattr(budget, "_ledger_dir", lambda: ledger)
    monkeypatch.delenv(budget.NESTED_WORKERS_ENV, raising=False)
    births = {100: "b100", 300: "b300-reborn", 400: "b400", me: "me"}
    monkeypatch.setattr(pl, "pid_is_alive", lambda pid: pid in births)
    monkeypatch.setattr(pl, "process_start_time", lambda pid: births.get(pid, ""))
    ledger.mkdir()
    for row in ({"pid": 100, "birth": "b100", "workers": 6, "command": "run_tests.py overlapped"},
                {"pid": 200, "birth": "gone", "workers": 4, "command": "dead"},
                {"pid": 300, "birth": "b300", "workers": 4, "command": "reborn pid"},
                {"pid": 400, "birth": "b400", "workers": 9, "command": "my own ancestor"}):
        (ledger / f"{row['pid']}.json").write_text(json.dumps(row), encoding="utf-8")
    (ledger / "garbage.json").write_text("{not json", encoding="utf-8")
    python, worker = "/opt/py/bin/python3", "import sys;exec(eval(sys.stdin.readline()))"
    table = [
        (400, 1, [python, "-m", "pytest", "tests/"]),  # the controller above this process: its own run
        (me, 400, [python, "-c", worker]),
        (100, 1, [python, "scripts/run_tests.py", "tests/"]),  # a leased owner...
        (101, 100, [python, "-m", "pytest", "tests/"]),  # ...whose tree the lease covers
        (700, 1, [python, "-m", "pytest", "-n", "8"]),  # an unleased root
        (701, 700, [python, "-c", worker]),
        (702, 701, [python, "-m", "pytest", "tests/test_nested.py"]),  # nested under that root: the same run
        (800, 1, ["/opt/node/bin/node", "--test", "a.test.js"]),  # the node lane of a bare run
        (801, 1, ["/opt/node/bin/node", "server.js"]),  # node without --test is no test run
        (900, 1, [python, "-I", "-S", "scripts/safe_test.py", "--", python, "-m", "pytest"]),  # stage one
        (901, 900, [python, "-I", "-S", "-c", "import sys; ...", "--", python, "-m", "pytest"]),  # stage two
        (902, 901, [python, "-m", "pytest", "tests/test_x.py"]),
    ]
    monkeypatch.setattr(budget, "_process_table", lambda: (table, ""))
    host = budget.other_runs()
    assert host == budget.HostRuns(slots=6, leases=((100, "run_tests.py overlapped"),), strangers=(
        (700, "python3 -m pytest -n 8"), (800, "node --test a.test.js"),
        (900, "python3 -I -S safe_test.py -- python3 -m pytest")))
    assert host.held == 9
    assert host.disclosure() == ("other test runs hold 6 slot(s): 100 run_tests.py overlapped; 3 unleased pytest/node "
                                 "root(s) counted one slot each: 700 python3 -m pytest -n 8, 800 node --test a.test.js, "
                                 "900 python3 -I -S safe_test.py -- python3 -m pytest")
    # A new lease unlinks the dead, reborn and unreadable rows, keeps the live ones and removes itself on close.
    lease = budget.SlotLease(3, "run_tests.py node")
    assert {path.name for path in ledger.glob("*.json")} == {"100.json", "400.json", f"{me}.json"}
    assert json.loads((ledger / f"{me}.json").read_text(encoding="utf-8")) == {
        "pid": me, "birth": "me", "workers": 3, "command": "run_tests.py node"}
    lease.close()
    assert sorted(path.name for path in ledger.glob("*.json")) == ["100.json", "400.json"]
    # A child with a parent allocation holds no row. An unreadable table is disclosed, never a silent zero,
    # and without it lineage is unknown: the ancestor's lease counts too (fewer workers, never more).
    monkeypatch.setenv(budget.NESTED_WORKERS_ENV, "1")
    assert budget.SlotLease(2).path is None and not (ledger / f"{me}.json").exists()
    monkeypatch.delenv(budget.NESTED_WORKERS_ENV)
    note = "process census unavailable (OSError): unleased runs were not counted"
    monkeypatch.setattr(budget, "_process_table", lambda: (None, note))
    assert budget.other_runs() == budget.HostRuns(15, ((100, "run_tests.py overlapped"), (400, "my own ancestor")), (), note)


def test_the_ledger_directory_is_this_users_own():
    from ouroboros.test_environment import default_temp_parent

    tag = str(os.getuid()) if hasattr(os, "getuid") else budget._user_tag()
    assert budget._ledger_dir() == default_temp_parent() / f"ouroboros-test-slots-{tag}.noindex"


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership, modes and links")
def test_a_planted_ledger_is_no_ledger_and_a_row_never_follows_a_link(tmp_path, monkeypatch):
    """F7 / astra 8: the ledger is this user's 0700 directory in the shared temp parent. A link, a file or another
    user's directory at its path is no ledger (nothing is read or written through it; the census still counts
    that user's runs as unleased roots), a directory of ours left wider is narrowed, and a row is staged under a
    fresh exclusive name and renamed over whatever sits at its own: a planted ``<pid>.json`` link is replaced,
    the old ``<pid>.json.tmp`` staging name is never opened, so no victim behind either is truncated."""
    ledger, me = tmp_path / "slots", os.getpid()
    monkeypatch.setattr(budget, "_ledger_dir", lambda: ledger)
    monkeypatch.delenv(budget.NESTED_WORKERS_ENV, raising=False)
    monkeypatch.setattr(budget, "_process_table", lambda: ([(700, 1, ["python3", "-m", "pytest"])], ""))
    births = {100: "b100", me: "me"}
    monkeypatch.setattr(pl, "pid_is_alive", lambda pid: pid in births)
    monkeypatch.setattr(pl, "process_start_time", lambda pid: births.get(pid, ""))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "100.json").write_text(json.dumps({"pid": 100, "birth": "b100", "workers": 6, "command": "planted"}))
    ledger.symlink_to(elsewhere, target_is_directory=True)
    assert budget._owned_ledger() is None
    assert budget.other_runs() == budget.HostRuns(strangers=((700, "python3 -m pytest"),))  # the planted row counts nothing
    assert budget.SlotLease(3, "x").path is None and sorted(path.name for path in elsewhere.iterdir()) == ["100.json"]
    ledger.unlink()
    ledger.write_text("not a directory", encoding="utf-8")
    assert budget._owned_ledger() is None
    ledger.unlink()
    ledger.mkdir()
    os.chmod(ledger, 0o755)
    assert budget._owned_ledger() == ledger and os.lstat(ledger).st_mode & 0o777 == 0o700
    real = os.getuid()
    with monkeypatch.context() as other_user:
        other_user.setattr(os, "getuid", lambda: real + 1)
        assert budget._owned_ledger() is None
    victim = tmp_path / "victim.json"
    victim.write_text("keep", encoding="utf-8")
    (ledger / f"{me}.json").symlink_to(victim)
    (ledger / f"{me}.json.tmp").symlink_to(victim)
    lease = budget.SlotLease(3, "run_tests.py x")
    row = ledger / f"{me}.json"
    assert lease.path == row and not row.is_symlink() and os.lstat(row).st_mode & 0o777 == 0o600
    assert json.loads(row.read_text(encoding="utf-8"))["workers"] == 3
    assert victim.read_text(encoding="utf-8") == "keep" and (ledger / f"{me}.json.tmp").is_symlink()
    # The exclusive staging file was renamed into the row; the lock file is the directory's only other entry.
    assert sorted(path.name for path in ledger.iterdir()) == [f"{me}.json", f"{me}.json.tmp", "ledger.lock"]
    lease.close()
    assert not row.exists()


def test_a_lease_narrows_to_what_a_pass_uses_and_never_widens(tmp_path, monkeypatch):
    """grok 7 / astra 14: the gate holds one row across both passes; at the one-process pass it holds one slot."""
    ledger, me = tmp_path / "slots", os.getpid()
    monkeypatch.setattr(budget, "_ledger_dir", lambda: ledger)
    monkeypatch.delenv(budget.NESTED_WORKERS_ENV, raising=False)
    monkeypatch.setattr(budget, "physical_memory", lambda: host_memory.MemorySnapshot(2 ** 40, 2 ** 40))
    monkeypatch.setattr(budget, "_process_table", lambda: ([], ""))
    admitted, lease = budget.reserve("hermetic gate", 4, slots=lambda admission: admission.workers + 5)
    row = lambda: json.loads((ledger / f"{me}.json").read_text(encoding="utf-8"))  # noqa: E731
    assert (admitted.workers, lease.workers, row()["workers"]) == (4, 4, 4)  # never more than admitted
    lease.hold(1)
    assert (lease.workers, row()["workers"], row()["command"]) == (1, 1, "hermetic gate")
    lease.hold(3)
    assert (lease.workers, row()["workers"]) == (1, 1)
    lease.close()
    _, one = budget.reserve("run_tests.py node", 4, slots=lambda admission: 2)
    assert (one.workers, row()["workers"]) == (2, 2)
    one.close()
    monkeypatch.setenv(budget.NESTED_WORKERS_ENV, "2")
    nested, lease = budget.reserve("child", 4)
    assert (nested, lease.path) == (budget.Admission(2, budget.HostRuns()), None)
    lease.hold(1)
    assert lease.workers == 1 and not list(ledger.glob("*.json"))


def test_two_runs_started_at_once_admit_under_one_ledger_lock(tmp_path, monkeypatch):
    """Astra 8: admission read the ledger and the row was written afterwards, so two runs started at once both
    saw an empty ledger and each took the whole machine. ``reserve`` holds the ledger's lock from the census to
    the row: a run arriving while another reserves waits the milliseconds that takes, then sees its row."""
    ledger = tmp_path / "slots"
    monkeypatch.setattr(budget, "_ledger_dir", lambda: ledger)
    monkeypatch.delenv(budget.NESTED_WORKERS_ENV, raising=False)
    assert budget._owned_ledger() == ledger
    child_code = textwrap.dedent(f"""
        import json, os, sys
        from pathlib import Path
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from ouroboros import host_memory, test_parallelism as budget
        os.environ.pop(budget.NESTED_WORKERS_ENV, None)  # the test seam gives every child one slot; this one is a run of its own
        ledger = Path({str(ledger)!r})
        budget._ledger_dir = lambda: ledger
        budget.physical_memory = lambda: host_memory.MemorySnapshot(2 ** 40, 2 ** 40)
        budget._process_table = lambda: ([], "")
        print("reserving", flush=True)
        admitted, lease = budget.reserve("child", 8)
        rows = {{json.loads(p.read_text())["pid"]: json.loads(p.read_text())["workers"] for p in ledger.glob("*.json")}}
        print(json.dumps({{"pid": os.getpid(), "workers": admitted.workers, "held": admitted.host.slots, "rows": rows}}), flush=True)
        lease.close()
    """)
    env = {key: value for key, value in os.environ.items() if key not in (budget.NESTED_WORKERS_ENV, budget.WORKERS_ENV)}
    with budget._ledger_lock(ledger):  # another run, mid-reservation
        child = subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.PIPE, text=True, env=env)
        assert child.stdout.readline().strip() == "reserving"
        time.sleep(0.5)  # the child is at the lock by now (or, without one, long past its admission)
        mine = budget.SlotLease(6, "parent", locked=True)
    try:
        out, _ = child.communicate(timeout=120)
    finally:
        mine.close()
    assert child.returncode == 0, out
    seen = json.loads(out.strip().splitlines()[-1])  # its own pid: a Windows venv's python.exe starts the interpreter as a child
    assert seen == {"pid": seen["pid"], "workers": 2, "held": 6, "rows": {str(os.getpid()): 6, str(seen["pid"]): 2}}


@pytest.mark.parametrize("capacity", [2, 3])
def test_the_gate_pins_the_worker_count_it_verified(tmp_path, monkeypatch, capacity):
    """The requested three slots still obey measured memory, not a hostile pytest override."""
    total = 20 * config.TEST_WORKER_PEAK_RSS_BYTES
    memory = host_memory.MemorySnapshot(total, int(total * config.HOST_MEMORY_RESERVE_FRACTION)
                                        + capacity * config.TEST_WORKER_PEAK_RSS_BYTES)
    monkeypatch.setattr(budget, "physical_memory", lambda: memory)
    monkeypatch.setenv(budget.WORKERS_ENV, "3")
    monkeypatch.setenv(budget.NESTED_WORKERS_ENV, "3")
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "1")
    env = preflight_runner._preflight_env(tmp_path / "root", tmp_path / "root" / "repo")
    assert env["PYTEST_XDIST_AUTO_NUM_WORKERS"] == str(capacity)
    assert budget.WORKERS_ENV not in env  # the candidate cannot re-lower the admission seam


def test_preflight_keeps_the_admission_it_measured_before_the_run(tmp_path, monkeypatch):
    from ouroboros import preflight_probe

    host = budget.HostRuns(slots=4, leases=((100, "run_tests.py overlapped"),), strangers=((700, "python3 -m pytest"),))
    monkeypatch.setattr(preflight_probe, "_preflight_admission", lambda: budget.Admission(2, host))
    preflight_runner.install_worker_probe(tmp_path)
    monkeypatch.setattr(preflight_probe, "_preflight_admission", lambda: budget.Admission(16, budget.HostRuns()))
    env = preflight_runner._preflight_env(tmp_path, tmp_path / "repo")
    assert env["PYTEST_XDIST_AUTO_NUM_WORKERS"] == "2"
    assert env[budget.NESTED_WORKERS_ENV] == "2"
    # The admission's facts travel with the run's evidence (`result.json` `workers`).
    assert preflight_runner.recorded_admission(tmp_path) == {
        "workers": 2, "held_by_other_runs": 4, "leases": [[100, "run_tests.py overlapped"]],
        "unleased_roots": [[700, "python3 -m pytest"]], "note": ""}


def test_local_node_lane_uses_the_same_cpu_and_memory_budget(tmp_path, monkeypatch):
    from ouroboros import preflight_node

    spec = importlib.util.spec_from_file_location("budget_test_runner", ROOT / "scripts/run_tests.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    peak = config.TEST_WORKER_PEAK_RSS_BYTES
    monkeypatch.setattr(budget, "physical_memory", lambda: host_memory.MemorySnapshot(10 * peak, 4 * peak))
    monkeypatch.setattr(budget, "other_runs", lambda: budget.HostRuns())  # a quiet host; the ledger has its own test
    monkeypatch.setenv(budget.WORKERS_ENV, "16")
    monkeypatch.delenv(budget.NESTED_WORKERS_ENV)
    monkeypatch.setattr(preflight_node, "candidate_node_tests", lambda repo: ["tests/a.test.js", "tests/b.test.js"])
    monkeypatch.setattr(preflight_node, "resolve_node", lambda: "/fake/node")
    monkeypatch.setattr(preflight_node, "probe_node_version", lambda node: "24.0.0")
    calls = []
    monkeypatch.setattr(runner, "_run", lambda label, argv, cwd: calls.append((argv, cwd)) or 0)
    assert runner._node_lane() == 0
    assert calls == [(["/fake/node", "--test", "--test-timeout=300000", "--test-concurrency=3", "tests/a.test.js",
                       "tests/b.test.js"],
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
