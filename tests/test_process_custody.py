"""Process custody: supervised spawn chokepoint, ledger, reaper, lifeline."""

import json
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
    "ouroboros/packaged_cli.py",          # user-facing CLI wrapper (foreground)
    "ouroboros/cli.py",                   # dev CLI (foreground)
    "ouroboros/server_control.py",        # restart exec path
    "ouroboros/headless.py",              # waited synchronous child
    "ouroboros/preflight_runner.py",      # waited hermetic pytest child
    "ouroboros/tools/shell.py",           # bounded foreground commands (waited + tracked)
    "ouroboros/tools/skill_exec.py",      # bounded skill run (waited + tracked)
    "ouroboros/tools/skill_preflight.py", # waited preflight child
    "ouroboros/marketplace/isolated_deps.py",  # waited installer child
    "ouroboros/gateways/claude_code.py",  # waited readonly child (timeout-bound)
    "ouroboros/extension_process_runner.py",  # waited extension child
    "ouroboros/workspace_executor.py",    # custody write-through added at spawn
    "ouroboros/local_model.py",           # custody record added at spawn
    "ouroboros/extension_companion.py",   # custody write-through added at spawn
    "ouroboros/tools/services.py",        # routed through spawn_supervised
}


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
