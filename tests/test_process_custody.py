"""Process custody: supervised spawn chokepoint, ledger, reaper, lifeline."""

import json
import multiprocessing
import os
import pathlib
import re
import subprocess
import sys
import time

import pytest

from ouroboros import process_custody
from ouroboros.process_custody import (
    ledger_path,
    reap_orphaned_processes,
    record_process,
    spawn_supervised,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Custody process-mechanics are POSIX-first (start_time/cmdline/pgid all
# degrade to liveness on Windows, where Job Objects are the primary kill
# mechanism). The spawn/reap tests deterministically wedge the Windows CI
# runner (suite-level KeyboardInterrupt at the same position on retry), so
# they run on POSIX only; the conformance scan below stays cross-platform.
_POSIX_ONLY = pytest.mark.skipif(
    os.name == "nt", reason="custody spawn/reap mechanics are POSIX-only"
)

# Popen call sites that legitimately bypass spawn_supervised: short-lived
# synchronous helpers (waited within the call), panic/cleanup layers, the
# launcher (custody host), and custody itself. Adding a NEW long-lived spawn
# site requires routing it through spawn_supervised/record_process or
# explicitly justifying it here.
_POPEN_ALLOWLIST = {
    "launcher.py",                        # custody host process (pre-runtime)
    "ouroboros/process_custody.py",       # the chokepoint itself
    "ouroboros/platform_layer.py",        # primitives (hidden_run helpers)
    # ProcessContainer.spawn IS the custody for the hermetic gate's short-lived
    # pytest root: membership is env-token/Job-Object-held and reaped at teardown,
    # so routing it through spawn_supervised would double-custody a bounded child.
    "ouroboros/process_containment.py",
    "ouroboros/packaged_cli.py",          # user-facing CLI wrapper (foreground)
    "ouroboros/cli.py",                   # dev CLI (foreground)
    "ouroboros/server_control.py",        # restart exec path
    "ouroboros/workspace_patch_capture.py",  # waited synchronous child
    "ouroboros/preflight_runner.py",      # waited hermetic pytest child
    "ouroboros/tools/shell_process.py",   # bounded foreground commands (waited + tracked)
    "ouroboros/tools/skill_exec.py",      # bounded skill run (waited + tracked)
    "ouroboros/tools/skill_preflight.py", # waited preflight child
    "ouroboros/marketplace/isolated_deps.py",  # waited installer child
    # Connect's vendor-CLI install (domain op, moved here by review wave 3): ONE
    # waited child spawned+registered atomically under the tools.shell lock,
    # so /panic's tracked-subprocess sweep can never observe it alive but
    # untracked (isolated_deps._run template).
    "ouroboros/claudexor_daemon.py",
    "ouroboros/extension_process_runner.py",  # waited extension child
    "ouroboros/workspace_executor.py",    # custody write-through added at spawn
    "ouroboros/local_model.py",           # custody record added at spawn
    "ouroboros/extension_companion.py",   # custody write-through added at spawn
    "ouroboros/tools/services.py",        # routed through spawn_supervised
    "supervisor/update_merge.py",        # bounded pre-restart import/compile smoke
    "supervisor/git_ops.py",             # shared bounded Git/dependency helpers (waited + panic-tracked)
    # v7 G1 split: sync_runtime_dependencies (the waited + panic-tracked pip
    # child) moved into the checkout/reset leaf with its custody unchanged.
    "supervisor/git_ops_reset.py",
    # D34 carrier engine: short-lived bounded git plumbing (waited, own process
    # group, whole-tree kill on timeout); kept self-contained so the standalone
    # operator rebase helper can import the module without the runtime stack.
    "supervisor/update_carriers.py",
    "ouroboros/colab_bootstrap.py",      # bounded Colab clone/fetch helper
}


def _spawn_service_from_worker_process(drive_root: str, result_queue) -> None:
    proc = spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        drive_root=pathlib.Path(drive_root),
        purpose="service:worker-owned",
        scope="session",
        owner_task_id="worker-task",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    result_queue.put(proc.pid)


def test_popen_sites_are_custodied_or_allowlisted():
    pattern = re.compile(r"subprocess\.Popen\(|[^.\w]Popen\(")
    offenders = []
    for base in ("ouroboros", "supervisor"):
        for path in (REPO_ROOT / base).rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            if pattern.search(text) and rel not in _POPEN_ALLOWLIST:
                offenders.append(rel)
    for name in ("server.py", "launcher.py"):
        path = REPO_ROOT / name
        if path.exists() and pattern.search(path.read_text(encoding="utf-8", errors="replace")):
            if name not in _POPEN_ALLOWLIST:
                offenders.append(name)
    assert not offenders, (
        "New raw Popen call sites outside the custody allowlist: "
        f"{offenders}. Route long-lived spawns through "
        "process_custody.spawn_supervised (or record_process write-through) "
        "or extend the allowlist with a justification comment."
    )


@_POSIX_ONLY
def test_spawn_supervised_records_ledger_entry(tmp_path):
    proc = spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        drive_root=tmp_path,
        purpose="test-sleeper",
        scope="task",
        owner_task_id="t123",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        lines = ledger_path(tmp_path).read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[-1])
        assert entry["pid"] == proc.pid
        assert entry["purpose"] == "test-sleeper"
        assert entry["scope"] == "task"
        assert entry["owner_task"] == "t123"
        assert entry["session_id"] == process_custody.current_custody_session_id()
        if os.name != "nt":
            assert entry["fingerprint"]["start_time"]
        assert entry["fingerprint"]["cmd_sha256"]
    finally:
        proc.kill()
        proc.wait(timeout=5)


@_POSIX_ONLY
def test_update_quiesce_kills_service_recorded_by_worker_process(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    worker = ctx.Process(
        target=_spawn_service_from_worker_process,
        args=(str(tmp_path), result_queue),
    )
    worker.start()
    service_pid = result_queue.get(timeout=20)
    worker.join(timeout=20)
    assert worker.exitcode == 0
    try:
        assert process_custody.pid_is_alive(service_pid)

        ok, blockers = process_custody.quiesce_custodied_services(tmp_path)

        assert ok is True
        assert blockers == []
        assert not process_custody.pid_is_alive(service_pid)
    finally:
        from ouroboros.platform_layer import kill_pid_tree

        kill_pid_tree(service_pid)


def test_update_quiesce_blocks_on_unreadable_custody_ledger(tmp_path):
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken\n", encoding="utf-8")

    ok, blockers = process_custody.quiesce_custodied_services(tmp_path)

    assert ok is False
    assert blockers == ["custody_ledger:unreadable"]


@_POSIX_ONLY
def test_reaper_kills_stale_session_entry(tmp_path, monkeypatch):
    proc = spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        drive_root=tmp_path,
        purpose="stale-service",
        scope="task",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Simulate a NEW server generation.
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
        reaped = reap_orphaned_processes(tmp_path)
        assert proc.pid in reaped
        deadline = time.time() + 5
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.05)
        assert proc.poll() is not None, "stale-session process must be dead"
        events = (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
        assert "process_reaped" in events
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_spawn_supervised_kills_child_when_custody_record_fails(tmp_path, monkeypatch):
    killed = []

    class FakeProc:
        pid = 4321

        def wait(self, timeout=None):
            return -9

    proc = FakeProc()
    monkeypatch.setattr(process_custody.subprocess, "Popen", lambda *_a, **_k: proc)
    monkeypatch.setattr(
        process_custody,
        "record_process",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("ledger full")),
    )
    monkeypatch.setattr(process_custody, "kill_process_tree", lambda value: killed.append(value))

    with pytest.raises(RuntimeError, match="durable custody"):
        spawn_supervised(
            [sys.executable, "-c", "pass"],
            drive_root=tmp_path,
            purpose="service:test",
            scope="task",
        )

    assert killed == [proc]


def test_update_quiesce_kills_service_group_that_outlives_leader(tmp_path, monkeypatch):
    entry = {
        "pid": 123,
        "pgid": 456,
        "purpose": "service:background-child",
        "scope": "session",
    }
    rewritten = []
    killed = []
    group_alive = {456: True}
    monkeypatch.setattr(process_custody, "_read_ledger_strict", lambda _root: (True, [entry]))
    monkeypatch.setattr(process_custody, "_fingerprint_matches", lambda _entry: False)
    monkeypatch.setattr(process_custody, "process_group_is_alive", lambda pgid: group_alive.get(pgid, False))
    monkeypatch.setattr(
        process_custody,
        "kill_process_group_id",
        lambda pgid: (killed.append(pgid), group_alive.__setitem__(pgid, False)),
    )
    monkeypatch.setattr(
        process_custody,
        "_rewrite_ledger",
        lambda _root, entries: rewritten.extend(entries),
    )

    ok, blockers = process_custody.quiesce_custodied_services(tmp_path)

    assert ok is True
    assert blockers == []
    assert killed == [456]
    assert rewritten == []


@_POSIX_ONLY
def test_reaper_never_kills_fingerprint_mismatch(tmp_path, monkeypatch):
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # Ledger entry claims this pid but with a FOREIGN fingerprint —
        # models a recycled pid from another install. Strict rule: never kill.
        record = record_process(
            tmp_path, pid=proc.pid, cmd=["sleep", "60"],
            purpose="foreign", scope="task",
        )
        entries = [dict(record, fingerprint={"start_time": "FOREIGN", "cmd_sha256": "deadbeef"})]
        process_custody._rewrite_ledger(tmp_path, entries)
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
        reaped = reap_orphaned_processes(tmp_path)
        assert proc.pid not in reaped
        assert proc.poll() is None, "fingerprint-mismatched process must survive"
    finally:
        proc.kill()
        proc.wait(timeout=5)


def _sleeper(tmp_path, purpose, scope, **kw):
    return spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        drive_root=tmp_path, purpose=purpose, scope=scope,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw,
    )


def _await_dead(proc):
    deadline = time.time() + 5
    while time.time() < deadline and proc.poll() is None:
        time.sleep(0.05)


@_POSIX_ONLY
def test_reaper_keeps_genuine_daemon_and_same_session(tmp_path):
    # A genuine launcher daemon (server_restart_fallback, NOT a skill companion)
    # and a live same-session service are both kept — even with no live-owner set.
    proc = _sleeper(tmp_path, "server_restart_fallback", "daemon")
    proc2 = _sleeper(tmp_path, "live-session-service", "session")
    try:
        reaped = reap_orphaned_processes(tmp_path)
        assert proc.pid not in reaped and proc2.pid not in reaped
        assert proc.poll() is None and proc2.poll() is None
    finally:
        for p in (proc, proc2):
            p.kill(); p.wait(timeout=5)


@_POSIX_ONLY
def test_reaper_companion_log_only_does_not_kill(tmp_path):
    # Default (log-only): companion of an uninstalled skill is NOT killed, but a
    # process_would_reap event is recorded for the safe first rollout.
    proc = _sleeper(tmp_path, "companion:gone_skill:worker", "daemon")
    try:
        reaped = reap_orphaned_processes(tmp_path, live_owner_skills={"other_skill"})  # gone_skill not in the installed set
        assert proc.pid not in reaped and proc.poll() is None
        import json as _json
        lines = (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()
        wr = [_json.loads(l) for l in lines if l.strip() and '"process_would_reap"' in l]
        assert wr, "expected a process_would_reap event"
        assert wr[-1]["pid"] == proc.pid
        assert wr[-1]["owner_skill"] == "gone_skill"
        assert wr[-1]["reason"] == "owner_uninstalled"
    finally:
        proc.kill(); proc.wait(timeout=5)


@_POSIX_ONLY
def test_reaper_kills_companion_of_uninstalled_skill_when_enforced(tmp_path):
    proc = _sleeper(tmp_path, "companion:gone_skill:worker", "daemon")
    try:
        reaped = reap_orphaned_processes(tmp_path, live_owner_skills={"other_skill"}, enforce_companion_reap=True)
        assert proc.pid in reaped
        _await_dead(proc)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait(timeout=5)


@_POSIX_ONLY
def test_reaper_empty_live_owner_skills_is_keep_all(tmp_path):
    # Defense-in-depth: an explicitly EMPTY live_owner_skills is normalized to
    # unknown (keep-all), NOT "every skill uninstalled" — even under enforce an
    # empty install set must never mass-reap companions (it can transiently mean
    # the skills dir was momentarily unreadable).
    proc = _sleeper(tmp_path, "companion:gone_skill:worker", "daemon")
    try:
        reaped = reap_orphaned_processes(tmp_path, live_owner_skills=set(), enforce_companion_reap=True)
        assert proc.pid not in reaped and proc.poll() is None
        sup = tmp_path / "logs" / "supervisor.jsonl"
        assert not sup.exists() or '"process_would_reap"' not in sup.read_text(encoding="utf-8")
    finally:
        proc.kill(); proc.wait(timeout=5)


@_POSIX_ONLY
def test_reaper_keeps_companion_of_installed_skill_same_session(tmp_path):
    # Installed (in live set) + same session → kept even under enforce (the live
    # supervisor owns it; killing would race a wanted process).
    proc = _sleeper(tmp_path, "companion:live_skill:worker", "daemon")
    try:
        reaped = reap_orphaned_processes(tmp_path, live_owner_skills={"live_skill"}, enforce_companion_reap=True)
        assert proc.pid not in reaped and proc.poll() is None
    finally:
        proc.kill(); proc.wait(timeout=5)


@_POSIX_ONLY
def test_reaper_kills_foreign_generation_companion_when_enforced(tmp_path, monkeypatch):
    # Companion of a STILL-installed skill but from a previous generation is a
    # stale duplicate (start() always re-spawns a fresh pid) → killed under enforce.
    proc = _sleeper(tmp_path, "companion:live_skill:worker", "daemon")
    try:
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
        reaped = reap_orphaned_processes(tmp_path, live_owner_skills={"live_skill"}, enforce_companion_reap=True)
        assert proc.pid in reaped
        _await_dead(proc)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait(timeout=5)


@_POSIX_ONLY
def test_reaper_keeps_companion_when_live_set_unknown(tmp_path, monkeypatch):
    # live_owner_skills=None (could not be computed) → keep, never mass-kill —
    # even a foreign-generation companion under enforce.
    proc = _sleeper(tmp_path, "companion:gone_skill:worker", "daemon")
    try:
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
        reaped = reap_orphaned_processes(tmp_path, live_owner_skills=None, enforce_companion_reap=True)
        assert proc.pid not in reaped and proc.poll() is None
    finally:
        proc.kill(); proc.wait(timeout=5)


@_POSIX_ONLY
def test_reaper_parses_companion_owner_as_middle_segment(tmp_path, monkeypatch):
    # Owner is the MIDDLE field; a colon in the companion NAME must not corrupt it.
    # Foreign generation + enforce: a CORRECT parse → owner "my_skill" (installed)
    # → reapable as a stale duplicate → KILLED. A wrong parse (empty owner) would
    # gate out (reapable=False) and survive — so killing here proves the owner was
    # parsed as the middle segment, not "job:42"/"42".
    proc = _sleeper(tmp_path, "companion:my_skill:job:42", "daemon")
    try:
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
        reaped = reap_orphaned_processes(tmp_path, live_owner_skills={"my_skill"}, enforce_companion_reap=True)
        assert proc.pid in reaped
        _await_dead(proc)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait(timeout=5)


@_POSIX_ONLY
def test_task_scope_reaped_when_owner_task_gone(tmp_path):
    proc = spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        drive_root=tmp_path,
        purpose="task-service",
        scope="task",
        owner_task_id="finished-task",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        reaped = reap_orphaned_processes(tmp_path, running_task_ids={"some-other-task"})
        assert proc.pid in reaped
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="lifeline is POSIX-only")
def test_lifeline_kills_child_when_parent_dies(tmp_path):
    # Parent spawns a child that starts the lifeline, then the parent exits.
    child_src = (
        "import sys; sys.path.insert(0, %r);"
        "from ouroboros.process_custody import start_parent_lifeline;"
        "start_parent_lifeline(poll_sec=0.2, label='test');"
        "import time; time.sleep(60)"
    ) % str(REPO_ROOT)
    parent_src = (
        "import subprocess, sys, pathlib;"
        f"child = subprocess.Popen([sys.executable, '-c', {child_src!r}]);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid));"
    )
    pid_file = tmp_path / "child_pid"
    subprocess.run(
        [sys.executable, "-c", parent_src, str(pid_file)],
        check=True, timeout=15,
    )
    child_pid = int(pid_file.read_text())
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            return  # lifeline fired
        time.sleep(0.2)
    try:
        os.kill(child_pid, 9)
    except ProcessLookupError:
        return
    raise AssertionError("child outlived dead parent despite lifeline")


def _process_gone(pid: int) -> bool:
    """True once ``pid`` is dead — an unreaped zombie counts as dead too."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
    return not state.strip() or state.strip().startswith("Z")


@pytest.mark.skipif(os.name == "nt", reason="lifeline is POSIX-only")
@pytest.mark.parametrize("start_method", ["forkserver", "spawn"])
def test_lifeline_fires_on_supervisor_death_under_every_start_method(tmp_path, start_method):
    """The lifeline watches the SUPERVISOR, not the immediate parent: under forkserver
    the parent is the forkserver process, which outlives a SIGKILLed supervisor for as
    long as any worker holds its alive pipe, so a ppid watch never fires and the orphan
    would keep running LLM rounds until the next boot."""
    script = tmp_path / "supervisor.py"
    script.write_text(
        "import multiprocessing as mp, pathlib, sys, time\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "def child(armed):\n"
        "    from ouroboros.process_custody import start_parent_lifeline\n"
        "    start_parent_lifeline(poll_sec=0.2, label='test')\n"
        "    pathlib.Path(armed).write_text('armed')\n"
        "    time.sleep(60)\n"
        "if __name__ == '__main__':\n"
        f"    proc = mp.get_context({start_method!r}).Process(target=child, args=(sys.argv[2],))\n"
        "    proc.start()\n"
        "    pathlib.Path(sys.argv[1]).write_text(str(proc.pid))\n"
        "    time.sleep(60)\n"
    )
    pid_file, armed = tmp_path / "child_pid", tmp_path / "armed"
    # Own session: the lifeline's group-kill can only ever hit this tree, and the
    # cleanup killpg below reaps the forkserver/resource-tracker helpers with it.
    supervisor = subprocess.Popen([sys.executable, str(script), str(pid_file), str(armed)], start_new_session=True)
    try:
        deadline = time.time() + 60
        while not armed.exists() and supervisor.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        assert armed.exists(), "child never armed its lifeline"
        child_pid = int(pid_file.read_text())
        supervisor.kill()
        supervisor.wait(timeout=10)
        deadline = time.time() + 15
        while time.time() < deadline and not _process_gone(child_pid):
            time.sleep(0.2)
        assert _process_gone(child_pid), f"{start_method} child outlived the dead supervisor"
    finally:
        try:
            os.killpg(supervisor.pid, 9)
        except ProcessLookupError:
            pass
        if supervisor.poll() is None:
            supervisor.wait(timeout=5)


# --- NW-10: custody session-id adoption + keep-service sparing ---

@pytest.mark.skipif(os.name == "nt", reason="POSIX session semantics")
def test_adopt_session_id_overrides_generation():
    original = process_custody.current_custody_session_id()
    try:
        process_custody.adopt_session_id("server-generation-xyz")
        assert process_custody.current_custody_session_id() == "server-generation-xyz"
        # Empty / whitespace values are ignored (never blanks the id).
        process_custody.adopt_session_id("")
        assert process_custody.current_custody_session_id() == "server-generation-xyz"
    finally:
        process_custody.adopt_session_id(original)
    assert process_custody.current_custody_session_id() == original


@pytest.mark.skipif(os.name == "nt", reason="POSIX kill semantics")
def test_kill_pid_tree_spares_excluded_pid():
    from ouroboros.platform_layer import kill_pid_tree
    keep = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    doomed = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        # Excluding `keep` must spare it even though we ask to kill its tree.
        kill_pid_tree(keep.pid, exclude_pids={keep.pid})
        time.sleep(0.5)
        assert keep.poll() is None, "excluded pid must survive kill_pid_tree"
        # A normal kill (no exclusion) terminates the process.
        kill_pid_tree(doomed.pid)
        doomed.wait(timeout=5)
        assert doomed.poll() is not None
    finally:
        for p in (keep, doomed):
            if p.poll() is None:
                p.kill()
                p.wait(timeout=5)


def test_live_kept_service_pids_reports_only_live_session_services(tmp_path):
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    keep = spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        drive_root=tmp_path, purpose="service:web", scope="session",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    task_svc = spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        drive_root=tmp_path, purpose="service:db", scope="task", owner_task_id="t1",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Local-executor keep-services use the workspace_service: prefix + session
    # scope; they must also be spared (NW-10 / review round-2 coverage).
    ws_keep = spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        drive_root=tmp_path, purpose="workspace_service:api", scope="session",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    ws_task = spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        drive_root=tmp_path, purpose="workspace_service:cache", scope="task", owner_task_id="t2",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        pids = process_custody.live_kept_service_pids(tmp_path)
        assert keep.pid in pids, "session-scope service must be reported as kept"
        assert ws_keep.pid in pids, "session-scope workspace_service must be reported as kept"
        assert task_svc.pid not in pids, "task-scope service is not a kept service"
        assert ws_task.pid not in pids, "task-scope workspace_service is not a kept service"
    finally:
        for p in (keep, task_svc, ws_keep, ws_task):
            p.kill()
            p.wait(timeout=5)


def test_installed_skill_names_is_disk_derived_and_none_on_failure(monkeypatch):
    # The reaper's owner-set source: disk-derived, and None (keep-all) on failure
    # OR empty disk — never an empty set() that would look like "all uninstalled".
    import server
    import ouroboros.skill_loader as skl
    from types import SimpleNamespace

    def _raise(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(skl, "discover_skills", _raise)
    assert server._installed_skill_names() is None  # raise → keep-all

    monkeypatch.setattr(skl, "discover_skills", lambda *a, **k: [])
    assert server._installed_skill_names() is None  # empty disk → keep-all (not set())

    monkeypatch.setattr(skl, "discover_skills",
                        lambda *a, **k: [SimpleNamespace(name="alpha"), SimpleNamespace(name="beta")])
    assert server._installed_skill_names() == {"alpha", "beta"}


# --- OB-06: /proc-first start times, matched dual-format for pre-upgrade rows ---
#
# These call ``_fingerprint_matches`` DIRECTLY on purpose. The reaper's cheap
# same-session liveness path would answer for a live session row before the
# fingerprint is ever computed, so a reaper-level assertion would pass without
# proving anything about the matcher.


def _fingerprint_entry(tmp_path, *, start_time=None):
    """A ledger entry for a real live process, with its start-time token overridden.

    ``start_time=None`` records the host's own legacy (``ps``) token, which is exactly
    what a pre-upgrade row on a Linux box would carry.
    """
    proc = _sleeper(tmp_path, "ob06-fingerprint", "task")
    record = record_process(
        tmp_path, pid=proc.pid, cmd=["sleep", "60"], purpose="ob06", scope="task",
    )
    if start_time is None:
        start_time = process_custody.process_start_time_legacy(proc.pid)
    fp = dict(record["fingerprint"], start_time=start_time)
    # The override simulates a PRE-change row: those carry no boot-qualified sibling
    # (the current writer's `start_time_boot` would otherwise satisfy the boot-first
    # equality and mask what the compatibility comparison under test actually does).
    fp.pop("start_time_boot", None)
    return proc, dict(record, fingerprint=fp)


@_POSIX_ONLY
def test_fingerprint_accepts_a_pre_upgrade_ps_start_time(tmp_path, monkeypatch):
    """A row written before /proc-first carries the ``ps -o lstart=`` spelling.

    After the upgrade the current token is boot-qualified, so recorded != current for
    every pre-existing row. Without the dual-format match the whole existing ledger
    would read as fingerprint-mismatched and be pruned on the next sweep.
    """
    if not process_custody.process_start_time_legacy(os.getpid()):
        pytest.skip("no usable ps/lstart start-time token on this host")
    proc, entry = _fingerprint_entry(tmp_path)
    try:
        assert entry["fingerprint"]["start_time"], "ps must answer for a live pid"
        # Simulate the post-upgrade Linux world regardless of the host we run on.
        monkeypatch.setattr(process_custody, "process_start_time", lambda pid: "4242.deadbeef")
        assert process_custody._fingerprint_matches(entry) is True
    finally:
        proc.kill(); proc.wait(timeout=5)


@_POSIX_ONLY
def test_fingerprint_refuses_a_bare_tick_row_on_a_ps_capable_host(tmp_path, monkeypatch):
    """A recorded BARE tick (pre-change ps-failed fallback) must never authorize a kill:
    ticks recur across reboots, so matching it against the tick half of a boot-qualified
    live token is exactly the cross-boot collision the boot id exists to refuse. On a
    ``ps``-capable host the row compares against the ``ps`` spelling, fails, and is
    PRUNED — the safe direction. (The one host class that ever MINTS bare ticks has no
    usable ``ps``; there ``process_start_time_legacy`` reproduces the bare tick and the
    row still resolves — the matrix pins that arm.)
    """
    proc, entry = _fingerprint_entry(tmp_path, start_time="4242")
    try:
        monkeypatch.setattr(process_custody, "process_start_time", lambda pid: "4242.deadbeef")
        monkeypatch.setattr(
            process_custody, "process_start_time_legacy", lambda pid: "Mon Aug 25 12:00:00 2026")
        assert process_custody._fingerprint_matches(entry) is False
    finally:
        proc.kill(); proc.wait(timeout=5)


@_POSIX_ONLY
def test_fingerprint_rejects_identical_ticks_from_another_boot(tmp_path, monkeypatch):
    """Boot-relative ticks RECUR across reboots — the boot id is what makes them an identity.

    Same tick count, different boot: a bare-tick token would have matched and authorized a
    kill of an unrelated process that merely inherited the pid. The qualified token must not.
    """
    proc, entry = _fingerprint_entry(tmp_path, start_time="4242.aaaaaaaa")
    try:
        monkeypatch.setattr(process_custody, "process_start_time", lambda pid: "4242.bbbbbbbb")
        assert process_custody._fingerprint_matches(entry) is False
    finally:
        proc.kill(); proc.wait(timeout=5)


@_POSIX_ONLY
def test_writer_keeps_the_legacy_spelling_downgrade_safe(tmp_path):
    """Rollback safety (the durable-format hazard): the unversioned ``start_time``
    field keeps the ``ps`` spelling an N−1 reader compares with its own
    ``process_start_time`` — so after a managed-update ROLLBACK every row written by
    this version still matches its live process (kill/prune reasoning intact) instead
    of silently pruning WITHOUT a kill and orphaning the process forever. The
    boot-qualified token rides the separate ``start_time_boot`` sibling, which an old
    reader simply ignores."""
    proc = _sleeper(tmp_path, "ob06-writer", "task")
    try:
        fp_row = record_process(
            tmp_path, pid=proc.pid, cmd=["sleep", "60"], purpose="ob06w", scope="task",
        )["fingerprint"]
        legacy = process_custody.process_start_time_legacy(proc.pid)
        assert fp_row["start_time"] == legacy, "the N-1 reader's comparison value"
        boot = process_custody.process_start_time(proc.pid)
        if boot and boot != legacy:  # Linux with a readable boot id
            assert fp_row.get("start_time_boot") == boot
        else:  # macOS/BSD: one spelling only, no redundant sibling
            assert "start_time_boot" not in fp_row
        # And the CURRENT reader accepts its own row (boot-first, no ps needed).
        assert process_custody._fingerprint_matches(
            {"pid": proc.pid, "fingerprint": fp_row}) is True
    finally:
        proc.kill(); proc.wait(timeout=5)


@_POSIX_ONLY
def test_sibling_free_row_falls_back_to_the_legacy_field(tmp_path, monkeypatch):
    """The REACHABLE mid-generation case: the boot id was unreadable at mint (so the
    row carries only the ``ps`` spelling, no boot sibling) and became readable by the
    sweep — one ``ps`` on this already-mismatched path keeps the row instead of
    pruning a live owned process."""
    real_ps = process_custody.process_start_time_legacy(os.getpid())
    if not real_ps or real_ps.isdigit():
        pytest.skip("no usable ps/lstart start-time token on this host")
    proc, entry = _fingerprint_entry(tmp_path, start_time=None)
    try:
        live_legacy = process_custody.process_start_time_legacy(proc.pid)
        fp = {"start_time": live_legacy}
        monkeypatch.setattr(process_custody, "process_start_time", lambda pid: "4242.aabbccdd")
        assert process_custody._fingerprint_matches(dict(entry, fingerprint=fp)) is True
    finally:
        proc.kill(); proc.wait(timeout=5)


@_POSIX_ONLY
def test_refuted_boot_evidence_never_requalifies_through_bare_ticks(tmp_path, monkeypatch):
    """A row minted on a ``ps``-less host carries a bare-tick ``start_time`` PLUS a
    boot-qualified sibling. After a reboot, the mismatched sibling is positive evidence
    of another boot; the legacy helper (which degrades to bare ticks there) must not
    re-qualify the cross-boot recycled pid — the row prunes, never kills."""
    proc, entry = _fingerprint_entry(tmp_path, start_time=None)
    try:
        fp = {"start_time": _B, "start_time_boot": _Q}
        monkeypatch.setattr(process_custody, "process_start_time", lambda pid: _Q2)
        monkeypatch.setattr(process_custody, "process_start_time_legacy", lambda pid: _B)
        assert process_custody._fingerprint_matches(dict(entry, fingerprint=fp)) is False
    finally:
        proc.kill(); proc.wait(timeout=5)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc start times are Linux-only")
def test_process_start_time_is_boot_qualified_on_linux():
    """On Linux the token comes from /proc and carries the boot id, without spawning ``ps``."""
    token = process_custody.process_start_time(os.getpid())
    assert re.fullmatch(r"\d+\.[0-9a-f]{1,32}", token), token
    assert token != process_custody.process_start_time_legacy(os.getpid())


@_POSIX_ONLY
def test_foreign_row_mismatch_costs_no_second_ps_without_proc(tmp_path, monkeypatch):
    """On a host with no /proc the current token already IS the ``ps`` spelling.

    Re-running ``ps`` there would return the byte-identical value that just failed to
    match, so the mismatch path must refuse before paying for a second subprocess.
    """
    proc, entry = _fingerprint_entry(tmp_path, start_time="FOREIGN")
    # The WRITE path legitimately pays one ``ps`` for the downgrade-safe legacy field;
    # the claim under test is about the SWEEP's mismatch path, so count from here.
    calls = []
    real_legacy = process_custody.process_start_time_legacy
    monkeypatch.setattr(
        process_custody, "process_start_time_legacy",
        lambda pid: (calls.append(pid), real_legacy(pid))[1],
    )
    monkeypatch.setattr(process_custody, "process_start_time", lambda pid: "Mon Aug 25 12:00:00 2026")
    try:
        assert process_custody._fingerprint_matches(entry) is False
        assert calls == [], f"a legacy-form current token must not trigger a second ps; got {calls}"
    finally:
        proc.kill(); proc.wait(timeout=5)


@_POSIX_ONLY
def test_unreadable_boot_id_prefers_the_collision_free_ps_token(monkeypatch):
    """With ticks but no boot id, ``ps`` wins — ``"<ticks>."`` is the LAST resort.

    Two ``"<ticks>."`` tokens from different boots string-MATCH, which is the exact
    cross-boot false positive this fix exists to prevent, and it sits on the kill path.
    The ``ps`` wall-clock token cannot collide across boots, so it is preferred; the
    separator form is reached only when ``ps`` has also failed.
    """
    from ouroboros import platform_layer

    monkeypatch.setattr(platform_layer, "_BOOT_ID", "")
    monkeypatch.setattr(platform_layer, "_proc_start_ticks", lambda pid: 4242)
    monkeypatch.setattr(
        pathlib.Path, "read_text",
        lambda self, *a, **k: (_ for _ in ()).throw(OSError("no boot id here")),
    )
    monkeypatch.setattr(platform_layer, "process_start_time_legacy", lambda pid: "Mon Aug 25 12:00:00 2026")
    assert platform_layer.process_start_time(os.getpid()) == "Mon Aug 25 12:00:00 2026"


@_POSIX_ONLY
def test_separator_form_is_the_last_resort_when_ps_also_fails(monkeypatch):
    """Only once ``ps`` has ALSO failed does the token degrade to ``"<ticks>."``.

    It still carries the separator so it can never be string-equal to a genuine
    pre-change bare-tick row — the ambiguity the separator removes.
    """
    from ouroboros import platform_layer

    monkeypatch.setattr(platform_layer, "_BOOT_ID", "")
    monkeypatch.setattr(platform_layer, "_proc_start_ticks", lambda pid: 4242)
    monkeypatch.setattr(
        pathlib.Path, "read_text",
        lambda self, *a, **k: (_ for _ in ()).throw(OSError("no boot id here")),
    )
    # ps failed, so the legacy helper degrades to its own bare-tick fallback.
    monkeypatch.setattr(platform_layer, "process_start_time_legacy", lambda pid: "4242")
    token = platform_layer.process_start_time(os.getpid())
    assert token == "4242.", token
    assert token != "4242", "a /proc token must never be confusable with a pre-change bare tick"


@_POSIX_ONLY
@pytest.mark.parametrize("boom", [
    OSError("transient"),
    UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),  # a ValueError: the fix-4 arm
])
def test_boot_id_read_failure_retries_then_latches(monkeypatch, boom):
    """One transient boot-id read failure must not downgrade the whole server generation.

    Both arms of the guard are covered: an OSError and a UnicodeDecodeError (a ValueError,
    which an unguarded caller would let escape and abort a whole custody sweep).
    """
    from ouroboros import platform_layer

    attempts = []

    def _read(self, *a, **k):
        attempts.append(str(self))
        if len(attempts) == 1:
            raise boom
        return "aabbccdd-0000-0000-0000-000000000000\n"

    monkeypatch.setattr(platform_layer, "_BOOT_ID", "")
    monkeypatch.setattr(platform_layer, "_proc_start_ticks", lambda pid: 7)
    monkeypatch.setattr(platform_layer, "process_start_time_legacy", lambda pid: "Mon Aug 25 12:00:00 2026")
    monkeypatch.setattr(pathlib.Path, "read_text", _read)

    # The failed read falls to the collision-free ps token, NOT to "7." (see the priority order).
    assert platform_layer.process_start_time(os.getpid()) == "Mon Aug 25 12:00:00 2026"
    assert platform_layer.process_start_time(os.getpid()) == "7.aabbccdd000000000000000000000000"   # retried, boot id now readable
    assert len(attempts) == 2
    assert platform_layer.process_start_time(os.getpid()) == "7.aabbccdd000000000000000000000000"   # latched
    assert len(attempts) == 2, "a successful read must latch and stop re-reading"


# The token forms a ledger row (or a live process) can carry, after the priority reorder:
#   Q  boot-qualified "<ticks>.<boot8>"      — post-change, boot id readable
#   Q2 same ticks, DIFFERENT boot            — the cross-boot impostor
#   P  ps wall-clock string                  — pre-change, or post-change without a boot id
#   S  "<ticks>."                            — post-change last resort (no boot id AND no ps)
#   B  bare "<ticks>"                        — pre-change ps-failed fallback only
_Q, _Q2, _P, _S, _B = "4242.aabbccdd", "4242.bbbbbbbb", "Mon Aug 25 12:00:00 2026", "4242.", "4242"


@_POSIX_ONLY
@pytest.mark.parametrize(("live", "recorded", "expected", "why"), [
    (_Q, _Q, True, "same boot, same ticks"),
    (_Q, _Q2, False, "SAME ticks from another boot must never match"),
    (_Q, _P, True, "pre-change ps row still matches after the upgrade"),
    (_Q, _B, False, "a bare tick carries no boot id and must never authorize a kill (prunes)"),
    (_Q, _S, False, "a row from a boot-id outage window mismatches after recovery (prunes, never kills)"),
    (_P, _P, True, "no boot id on either side: the ps token is the identity"),
    (_P, _Q, False, "a boot-qualified row mismatches once the boot id stops being readable"),
    (_P, _B, False, "a bare-tick row against a ps-form live token prunes (never a tick-half kill)"),
    (_S, _S, True, "the disclosed last-resort collision: both sides degraded identically"),
    (_S, _B, True, "pre-change bare tick still matches the tick half of the last-resort form"),
    (_S, _P, False, "ps is broken here, so a ps-format row cannot be re-derived"),
])
def test_start_time_match_matrix(tmp_path, monkeypatch, live, recorded, expected, why):
    """Every (live form x recorded form) pair the priority reorder can produce."""
    # ``ps`` answers only while the live token is not itself a ps string; when the live token
    # IS ps-shaped, re-running ps returns that same value (which is what fix 1 short-circuits).
    monkeypatch.setattr(process_custody, "process_start_time_legacy",
                        lambda pid: _P if live in (_Q, _P) else _B)
    monkeypatch.setattr(process_custody, "process_start_time", lambda pid: live)
    proc, entry = _fingerprint_entry(tmp_path, start_time=recorded)
    try:
        assert process_custody._fingerprint_matches(entry) is expected, why
    finally:
        proc.kill(); proc.wait(timeout=5)


# --- OB-07: cheap liveness for the current generation's own session rows ---


@_POSIX_ONLY
@pytest.mark.parametrize("purpose", ["live-session-service", "service:web", "workspace_service:api"])
def test_reaper_skips_the_fingerprint_for_live_same_session_rows(tmp_path, monkeypatch, purpose):
    """A live same-session session row is kept either way, so the `ps` is pure cost.

    The counter is the assertion: worker-pool members, the SyncManager, the claudexor
    daemon, the local-model server and keep-services are ALL scope="session", so this is
    the hot majority of the ledger on every 600s tick and startup sweep.
    """
    calls = []
    real = process_custody._fingerprint_matches
    monkeypatch.setattr(
        process_custody, "_fingerprint_matches",
        lambda entry: (calls.append(entry.get("pid")), real(entry))[1],
    )
    proc = _sleeper(tmp_path, purpose, "session")
    try:
        reaped = reap_orphaned_processes(tmp_path)
        assert proc.pid not in reaped
        assert proc.poll() is None, "a live same-session service must be kept"
        assert calls == [], f"no fingerprint should be computed for a live session row; got {calls}"
        # and the row is still in the ledger for the next generation to judge
        kept = [json.loads(line) for line in ledger_path(tmp_path).read_text().splitlines() if line.strip()]
        assert [e["pid"] for e in kept] == [proc.pid]
    finally:
        proc.kill(); proc.wait(timeout=5)


def test_reaper_keeps_dead_leader_session_service_with_a_live_group(tmp_path, monkeypatch):
    """A DEAD pid must fall through to the group evidence, never take a prune shortcut.

    A session service whose leader died while its process group is still alive is
    preserved by ``_service_group_survives_leader``; pruning on a dead pid inside the
    cheap path would have silently dropped that row and orphaned the group.
    """
    entry = {
        "pid": 123,
        "pgid": 456,
        "purpose": "service:background-child",
        "scope": "session",
        "session_id": process_custody._SESSION_ID,  # CURRENT generation
        "fingerprint": {"start_time": "gone", "cmd_sha256": "gone"},
    }
    rewritten = []
    monkeypatch.setattr(process_custody, "_read_ledger", lambda _root: [entry])
    monkeypatch.setattr(process_custody, "pid_is_alive", lambda _pid: False)  # leader is dead
    monkeypatch.setattr(process_custody, "process_group_is_alive", lambda pgid: pgid == 456)
    monkeypatch.setattr(
        process_custody, "kill_process_group_id",
        lambda pgid: pytest.fail(f"a surviving service group must not be killed (pgid={pgid})"),
    )
    monkeypatch.setattr(process_custody, "_rewrite_ledger", lambda _root, entries: rewritten.extend(entries))

    assert reap_orphaned_processes(tmp_path) == []
    assert rewritten == [entry], "the dead-leader row must survive on its group evidence"
