"""Launcher-owned reaping of same-install stray server processes.

No test in this file signals a real process: the signal and descendant-discovery
seams are always spied, and process enumeration is always faked.
"""

import inspect
import logging
import os
import re
import sys
import types

import pytest

from ouroboros import launcher_server_reaper as reaper
from ouroboros import process_containment as containment

REPO = os.path.normpath("/opt/Ouroboros/repo")
DATA = os.path.normpath("/opt/Ouroboros/data")
OURS = f"{os.path.normpath('/opt/Ouroboros/python/bin/python3')} {os.path.join(REPO, 'server.py')}"


def _install_fakes(monkeypatch, pids, commands, env_states, groups=None):
    """Fake the three live-state readers the finder uses.

    ``env_states`` maps pid -> {(key, value): state}; a missing entry answers
    ABSENT, which (like UNREADABLE) is not a proof.
    """
    monkeypatch.setattr(
        reaper, "_candidate_commands", lambda: {pid: commands.get(pid, "") for pid in pids},
    )
    # Revalidation re-reads the live command through the platform layer.
    monkeypatch.setattr(
        "ouroboros.platform_layer.process_command",
        lambda pid: commands.get(pid, ""),
    )
    monkeypatch.setattr(
        "ouroboros.platform_layer.process_group_id",
        lambda pid: (groups or {}).get(pid, pid),
    )
    monkeypatch.setattr(
        reaper, "pid_environment_assignment_state",
        lambda pid, key, value: env_states.get(pid, {}).get(
            (key, value), containment.ENV_ASSIGNMENT_ABSENT
        ),
    )
    monkeypatch.setattr("ouroboros.platform_layer.IS_WINDOWS", False)
    # No getpid/getppid stubbing: `reaper.os` IS the interpreter-wide os module,
    # so patching it would leak into every other caller in-process. Fake pids
    # instead sit far above any OS pid range (Linux pid_max 4194304), so they
    # can never collide with the runner's real identity.


def _proof(data_dir=DATA):
    return {
        (reaper.MANAGED_MARKER_ENV, reaper.MANAGED_MARKER_VALUE): (
            containment.ENV_ASSIGNMENT_PRESENT
        ),
        (reaper.DATA_DIR_ENV, data_dir): containment.ENV_ASSIGNMENT_PRESENT,
    }


# ---------------------------------------------------------------------------
# Finder: what counts as proof
# ---------------------------------------------------------------------------

def test_both_env_assignments_are_required_before_a_pid_is_proven(monkeypatch):
    """The marker alone, or the data dir alone, is not a licence to kill: a
    direct run of this checkout that exported our data dir is still not managed."""
    _install_fakes(
        monkeypatch,
        pids=[9990101, 9990102, 9990103],
        commands={9990101: OURS, 9990102: OURS, 9990103: OURS},
        env_states={
            9990101: _proof(),
            9990102: {(reaper.DATA_DIR_ENV, DATA): containment.ENV_ASSIGNMENT_PRESENT},
            9990103: {
                (reaper.MANAGED_MARKER_ENV, reaper.MANAGED_MARKER_VALUE): (
                    containment.ENV_ASSIGNMENT_PRESENT
                )
            },
        },
    )
    proven, unproven = reaper.find_same_install_server_pids(REPO, DATA)
    assert proven == [9990101]
    assert sorted(unproven) == [9990102, 9990103]


def test_an_unreadable_environment_is_spared_not_killed(monkeypatch):
    """macOS `ps -E` omits an environment it may not read, and /proc returns
    EACCES for a nondumpable process. Unanswered is never answered-yes."""
    _install_fakes(
        monkeypatch,
        pids=[9990201],
        commands={9990201: OURS},
        env_states={
            9990201: {
                (reaper.MANAGED_MARKER_ENV, reaper.MANAGED_MARKER_VALUE): (
                    containment.ENV_ASSIGNMENT_UNREADABLE
                ),
                (reaper.DATA_DIR_ENV, DATA): containment.ENV_ASSIGNMENT_UNREADABLE,
            }
        },
    )
    proven, unproven = reaper.find_same_install_server_pids(REPO, DATA)
    assert proven == [] and unproven == [9990201]


def test_a_different_data_directory_is_not_our_generation(monkeypatch):
    _install_fakes(
        monkeypatch,
        pids=[9990301],
        commands={9990301: OURS},
        env_states={9990301: _proof(data_dir="/opt/Ouroboros/other-data")},
    )
    proven, unproven = reaper.find_same_install_server_pids(REPO, DATA)
    assert proven == [] and unproven == [9990301]


def test_sibling_installs_and_dev_clones_are_never_enumerated(monkeypatch):
    """A second packaged install and a development checkout run a DIFFERENT
    server.py path, so they are not candidates however their env reads."""
    sibling = "/Users/o/Ouroboros/python/bin/python3 /Users/o/Ouroboros/repo/server.py"
    dev = "/usr/bin/python3 /home/dev/src/ouroboros/server.py"
    _install_fakes(
        monkeypatch,
        pids=[9990401, 9990402],
        commands={9990401: sibling, 9990402: dev},
        env_states={9990401: _proof(), 9990402: _proof()},
    )
    proven, unproven = reaper.find_same_install_server_pids(REPO, DATA)
    assert proven == [] and unproven == []


def test_self_parent_caller_exclusions_and_known_groups_are_skipped(monkeypatch):
    """The launcher, its parent, an explicitly excluded pid and anything sharing
    a known process group are part of a tree we already account for."""
    me, parent = os.getpid(), os.getppid()
    _install_fakes(
        monkeypatch,
        pids=[me, parent, 9990501, 9990502, 9990503],
        commands={pid: OURS for pid in (me, parent, 9990501, 9990502, 9990503)},
        env_states={pid: _proof() for pid in (me, parent, 9990501, 9990502, 9990503)},
        # 9990502 shares the caller-excluded pid's group; 9990503 stands alone.
        groups={me: me, parent: parent, 9990501: 9990501, 9990502: 9990501, 9990503: 9990503},
    )
    proven, unproven = reaper.find_same_install_server_pids(
        REPO, DATA, exclude_pids=[9990501],
    )
    assert proven == [9990503] and unproven == []


def test_candidate_enumeration_uses_one_unbranded_full_width_ps_read(monkeypatch):
    """ps, not a pattern-scoped pgrep: candidate selection must not depend on
    the install path containing any particular word (REPO_DIR is configurable);
    -u scopes to this user, -ww defeats BSD truncation."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return types.SimpleNamespace(
            stdout="  777 python3 /x/server.py\nnot-a-pid oops\n", returncode=0,
        )

    monkeypatch.setattr(reaper, "os", types.SimpleNamespace(getuid=lambda: 501))
    monkeypatch.setattr(reaper.subprocess, "run", fake_run)
    assert reaper._candidate_commands() == {777: "python3 /x/server.py"}
    assert seen["cmd"] == ["ps", "-ww", "-u", "501", "-o", "pid=,command="]


def test_a_missing_ps_is_enumeration_failure_not_a_clean_answer(monkeypatch):
    """No ps, or a ps error (nonzero rc), must be distinguishable from an empty
    result: nothing was checked."""
    def missing_run(cmd, **kwargs):
        raise FileNotFoundError("ps")

    monkeypatch.setattr(reaper.subprocess, "run", missing_run)
    assert reaper._candidate_commands() is None

    def erroring_run(cmd, **kwargs):
        return types.SimpleNamespace(stdout="", returncode=1)

    monkeypatch.setattr(reaper.subprocess, "run", erroring_run)
    assert reaper._candidate_commands() is None


def test_reap_names_an_enumeration_failure_instead_of_reading_clean(monkeypatch, caplog):
    monkeypatch.setattr("ouroboros.platform_layer.IS_WINDOWS", False)
    monkeypatch.setattr(reaper, "_env_proof_available", lambda: True)
    monkeypatch.setattr(reaper, "_candidate_commands", lambda: None)
    with caplog.at_level(logging.WARNING, logger=reaper.log.name):
        assert reaper.reap_same_install_strays(REPO, DATA) == []
    assert any("could not enumerate" in r.getMessage() for r in caplog.records)


def test_a_whitespace_repo_path_disables_the_sweep_with_one_named_warning(monkeypatch, caplog):
    """The exact-token proof cannot represent a whitespace path; silence would
    hide that such an install is never swept."""
    monkeypatch.setattr("ouroboros.platform_layer.IS_WINDOWS", False)
    monkeypatch.setattr(reaper, "_env_proof_available", lambda: True)
    with caplog.at_level(logging.WARNING, logger=reaper.log.name):
        assert reaper.reap_same_install_strays("/opt/Our Oboros/repo", DATA) == []
    assert any("whitespace" in r.getMessage() for r in caplog.records)
    assert reaper.find_same_install_server_pids("/opt/Our Oboros/repo", DATA) == ([], [])


def test_process_command_requests_unlimited_width(monkeypatch):
    """BSD ps truncates without -ww, and the identity rule matches exact argv
    tokens — a packaged interpreter path is long enough to push the server.py
    argument off a truncated line."""
    from ouroboros import platform_layer

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return types.SimpleNamespace(stdout="python3 /x/server.py\n", returncode=0)

    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)
    monkeypatch.setattr(platform_layer.subprocess, "run", fake_run)
    assert platform_layer.process_command(1234).endswith("server.py")
    assert "-ww" in seen["cmd"]


def test_the_finder_never_reads_the_custody_ledger():
    """Missing ledger entries are the defect being repaired, so consulting the
    ledger would spare exactly the strays that matter."""
    source = inspect.getsource(reaper)
    assert "process_ledger" not in source
    assert "process_custody" not in source


# ---------------------------------------------------------------------------
# Reap: kill mechanics
# ---------------------------------------------------------------------------

def _spy_kill(monkeypatch):
    killed = []
    # The enforcement gate needs a byte-exact env source; the test host may
    # not have /proc, and no test reads real kernel state anyway.
    monkeypatch.setattr(reaper, "_env_proof_available", lambda: True)
    monkeypatch.setattr(
        "ouroboros.platform_layer.collect_descendant_pids", lambda pid: [],
    )
    # The proven root and the descendants captured before it are signalled directly.
    monkeypatch.setattr(reaper, "_signal_pid", lambda pid: killed.append(pid))
    # Confirmed-dead follows the signal: liveness is answered from the spy so
    # no test outcome depends on which real pids exist on this machine.
    monkeypatch.setattr(reaper, "_pid_gone", lambda pid: pid in killed)
    monkeypatch.setattr(reaper, "_SETTLE_SEC", 0)
    monkeypatch.setattr(reaper, "_CONFIRM_DEADLINE_SEC", 0)
    return killed


def test_proven_strays_are_tree_killed_and_unproven_ones_are_spared(monkeypatch, caplog):
    killed = _spy_kill(monkeypatch)
    _install_fakes(
        monkeypatch,
        pids=[9990601, 9990602],
        commands={9990601: OURS, 9990602: OURS},
        env_states={9990601: _proof()},
    )
    with caplog.at_level(logging.WARNING, logger=reaper.log.name):
        survivors = reaper.reap_same_install_strays(REPO, DATA, "startup")
    # Three bounded passes, each re-proving the pid that never dies in this fake.
    assert set(killed) == {9990601}
    assert killed.count(9990601) == reaper.REAP_PASSES
    assert survivors == [9990601]
    assert 9990602 not in killed
    assert any("9990602" in record.getMessage() for record in caplog.records)


def test_a_pid_that_dies_between_proof_and_signal_is_not_killed(monkeypatch):
    """Revalidation happens immediately before the signal; a pid whose command
    line no longer matches may have been recycled onto a stranger."""
    killed = _spy_kill(monkeypatch)
    commands = {9990701: OURS}
    _install_fakes(
        monkeypatch, pids=[9990701], commands=commands, env_states={9990701: _proof()},
    )

    real_revalidate = reaper._revalidate_and_kill

    def vanishing(pid, server_paths, data_dir_values, retained_descendant_roots=None):
        commands[pid] = ""  # exited between the scan and the signal
        return real_revalidate(pid, server_paths, data_dir_values)

    monkeypatch.setattr(reaper, "_revalidate_and_kill", vanishing)
    assert reaper.reap_same_install_strays(REPO, DATA) == []
    assert killed == []


def test_a_fork_between_passes_is_caught_by_the_rescan(monkeypatch):
    """A stray forking mid-sweep hands its child the same cmdline and the same
    inherited environment, so the child is proven on the next pass."""
    killed = []
    live = {9990801: OURS}
    monkeypatch.setattr(reaper, "_candidate_commands", lambda: dict(live))
    monkeypatch.setattr(
        "ouroboros.platform_layer.process_command", lambda pid: live.get(pid, ""),
    )
    monkeypatch.setattr("ouroboros.platform_layer.process_group_id", lambda pid: pid)
    monkeypatch.setattr(
        reaper, "pid_environment_assignment_state",
        lambda pid, key, value: (
            containment.ENV_ASSIGNMENT_PRESENT if pid in live
            else containment.ENV_ASSIGNMENT_ABSENT
        ),
    )
    monkeypatch.setattr("ouroboros.platform_layer.IS_WINDOWS", False)
    monkeypatch.setattr(reaper, "_env_proof_available", lambda: True)
    monkeypatch.setattr(
        "ouroboros.platform_layer.collect_descendant_pids", lambda pid: [],
    )
    monkeypatch.setattr(reaper, "_pid_gone", lambda pid: pid not in live)
    monkeypatch.setattr(reaper, "_SETTLE_SEC", 0)
    monkeypatch.setattr(reaper, "_CONFIRM_DEADLINE_SEC", 0)

    def forking_signal(pid):
        killed.append(pid)
        live.pop(pid, None)
        if pid == 9990801:
            live[9990802] = OURS  # the fork inherits everything

    monkeypatch.setattr(reaper, "_signal_pid", forking_signal)
    survivors = reaper.reap_same_install_strays(REPO, DATA)
    assert killed == [9990801, 9990802]
    assert survivors == []


def test_the_sweep_is_bounded_and_reports_survivors(monkeypatch):
    """A pid that refuses to die must not become an unbounded kill loop."""
    killed = _spy_kill(monkeypatch)
    _install_fakes(
        monkeypatch, pids=[9990901], commands={9990901: OURS}, env_states={9990901: _proof()},
    )
    assert reaper.reap_same_install_strays(REPO, DATA) == [9990901]
    assert len(killed) == reaper.REAP_PASSES


def test_windows_sweeps_nothing(monkeypatch):
    """The kill-on-close Job Object already reaps orphans there."""
    killed = _spy_kill(monkeypatch)
    monkeypatch.setattr("ouroboros.platform_layer.IS_WINDOWS", True)
    assert reaper.reap_same_install_strays(REPO, DATA) == []
    assert reaper.find_same_install_server_pids(REPO, DATA) == ([], [])
    assert killed == []


# ---------------------------------------------------------------------------
# Launcher wiring
# ---------------------------------------------------------------------------

def test_the_sweep_runs_per_generation_after_record_cleanup_and_no_port_sweep(monkeypatch):
    """Behavioural: the per-generation helper runs the record cleanup, then the stray
    sweep, and hands BOTH kinds of unresolved PIDs to the caller; no port is swept."""
    import launcher

    calls: list = []
    monkeypatch.setattr(
        launcher, "_cleanup_recorded_server_process", lambda reason: calls.append(("record", reason)) or [17]
    )
    monkeypatch.setattr(
        launcher, "_reap_same_install_strays",
        lambda reason: calls.append(("sweep", reason)) or [4242, 17],
    )

    unresolved = launcher._pre_generation_cleanup()

    assert calls == [("record", "startup"), ("sweep", "startup")]
    assert unresolved == [17, 4242]
    # The lifecycle loop consumes exactly this helper before any start.
    loop_src = inspect.getsource(launcher.agent_lifecycle_loop)
    assert loop_src.index("_pre_generation_cleanup()") < loop_src.index("proc = start_agent(port)")


def test_a_proven_survivor_skips_start_agent_for_that_generation(monkeypatch):
    """Behavioural: with an unresolved predecessor reported, the generation must not start —
    the loop logs, waits, and comes back around instead of calling start_agent."""
    import threading

    import launcher

    started: list = []
    shutdown = threading.Event()

    def fake_cleanup():
        # First generation reports a survivor; ending the loop here keeps the
        # test to exactly one iteration.
        shutdown.set()
        return [4242]

    monkeypatch.setattr(launcher, "_shutdown_event", shutdown)
    monkeypatch.setattr(launcher, "_pre_generation_cleanup", fake_cleanup)
    monkeypatch.setattr(launcher, "start_agent", lambda port: started.append(port))
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)

    launcher.agent_lifecycle_loop(1)

    assert started == [], "an unresolved predecessor must suppress start_agent"


def test_main_leaves_predecessor_cleanup_to_the_lifecycle_loop():
    """main() runs no preflight copy of its own: each generation's loop resolves its predecessor.

    main() may still stop the generation its own startup recorded when that startup is quit,
    aborted or fails; those calls name their teardown reason, never the preflight's "startup"."""
    import launcher

    src = inspect.getsource(launcher.main)
    assert "_pre_generation_cleanup(" not in src and "_reap_same_install_strays(" not in src
    reasons = set(re.findall(r"_cleanup_recorded_server_process\(([^)]*)\)", src))
    assert reasons <= {'"startup_quit"', '"startup_abort"', '"startup_failure"'}, reasons


def test_the_panic_and_window_close_path_performs_no_stray_sweep():
    """BIBLE Emergency Stop: the panic path tears down what it owns and gains no
    new killing."""
    import launcher

    assert "_reap_same_install_strays" not in inspect.getsource(
        launcher._kill_orphaned_children
    )


# ---------------------------------------------------------------------------
# Env-assignment primitive
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX /proc and ps -E semantics")
def test_env_assignment_state_answers_for_our_own_pid():
    """Smoke test against live kernel state: an assignment this process cannot be
    carrying must never come back PRESENT, on either the /proc or the `ps -E` path."""
    state = containment.pid_environment_assignment_state(
        os.getpid(), "OUROBOROS_REAPER_PROBE_UNSET", "unit-test-placeholder",
    )
    assert state in (
        containment.ENV_ASSIGNMENT_ABSENT, containment.ENV_ASSIGNMENT_UNREADABLE,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX /proc and ps -E semantics")
def test_env_assignment_state_never_answers_present_for_an_impossible_pid():
    assert containment.pid_environment_assignment_state(
        -1, "OUROBOROS_REAPER_PROBE_UNSET", "unit-test-placeholder",
    ) != containment.ENV_ASSIGNMENT_PRESENT


def test_env_assignment_state_is_unreadable_on_windows(monkeypatch):
    monkeypatch.setattr("ouroboros.platform_layer.IS_WINDOWS", True)
    assert containment.pid_environment_assignment_state(
        os.getpid(), "PATH", os.environ.get("PATH", ""),
    ) == containment.ENV_ASSIGNMENT_UNREADABLE


def test_a_command_merely_mentioning_the_server_path_is_not_a_candidate(monkeypatch):
    """Substring hits are not identity: an editor or log tool naming the path, or
    a longer filename sharing the prefix, is neither killable nor spared-listed —
    only the launcher's `<python> <repo>/server.py` spawn shape is this
    install's server."""
    _install_fakes(
        monkeypatch,
        pids=[9990621, 9990622, 9990623],
        commands={
            9990621: f"vim {REPO}/server.py",
            9990622: f"/opt/Ouroboros/python/bin/python3 {REPO}/server.py.bak",
            9990623: f"tail -f {REPO}/server.py.log",
        },
        env_states={pid: _proof() for pid in (9990621, 9990622, 9990623)},
    )
    assert reaper.find_same_install_server_pids(REPO, DATA) == ([], [])


def test_a_sweep_aborted_mid_work_reports_survivors_not_clean(monkeypatch, caplog):
    """An exception after a pid was proven must not read as swept-clean: the
    caller would start the exact colliding generation the sweep exists to
    prevent."""
    killed = _spy_kill(monkeypatch)
    _install_fakes(
        monkeypatch, pids=[9990631], commands={9990631: OURS}, env_states={9990631: _proof()},
    )

    def exploding(pid, server_paths, data_dir_values, retained_descendant_roots=None):
        raise RuntimeError("mid-sweep failure")

    monkeypatch.setattr(reaper, "_revalidate_and_kill", exploding)
    with caplog.at_level(logging.WARNING, logger=reaper.log.name):
        assert reaper.reap_same_install_strays(REPO, DATA) == [9990631]
    assert killed == []
    assert any("aborted mid-work" in r.getMessage() for r in caplog.records)


def test_a_signalled_pid_still_alive_is_a_survivor_not_a_kill(monkeypatch, caplog):
    """The platform signal primitive swallows per-pid errors, so only a
    liveness read can say what the signal achieved; a pid logged as reaped
    while it survived would contradict the survivor report from the same generation."""
    signalled = []
    monkeypatch.setattr(reaper, "_env_proof_available", lambda: True)
    monkeypatch.setattr(
        "ouroboros.platform_layer.collect_descendant_pids", lambda pid: [],
    )
    monkeypatch.setattr(reaper, "_signal_pid", lambda pid: signalled.append(pid))
    monkeypatch.setattr(reaper, "_pid_gone", lambda pid: False)
    monkeypatch.setattr(reaper, "_SETTLE_SEC", 0)
    monkeypatch.setattr(reaper, "_CONFIRM_DEADLINE_SEC", 0)
    _install_fakes(
        monkeypatch, pids=[9990641], commands={9990641: OURS}, env_states={9990641: _proof()},
    )
    with caplog.at_level(logging.INFO, logger=reaper.log.name):
        assert reaper.reap_same_install_strays(REPO, DATA) == [9990641]
    assert signalled and all(pid == 9990641 for pid in signalled)
    assert not any("Reaped" in r.getMessage() for r in caplog.records)


def test_without_proc_the_sweep_is_report_only_with_a_named_warning(monkeypatch, caplog):
    """ps -E mixes argv into the environment column, so a KEY=value argv token
    reads as an assignment — argv must never authorize a kill."""
    monkeypatch.setattr("ouroboros.platform_layer.IS_WINDOWS", False)
    monkeypatch.setattr(reaper, "_env_proof_available", lambda: False)
    with caplog.at_level(logging.WARNING, logger=reaper.log.name):
        assert reaper.reap_same_install_strays(REPO, DATA) == []
    assert any("report-only" in r.getMessage() for r in caplog.records)


def test_descendants_are_captured_before_the_root_signal(monkeypatch):
    """SIGKILLing the root reparents its children to init, after which no
    parent-walk can find them: capture must precede the signal, and the
    captured children must be signalled too."""
    order = []
    monkeypatch.setattr(
        "ouroboros.platform_layer.collect_descendant_pids",
        lambda pid: order.append(("enum", pid)) or [9990902],
    )
    monkeypatch.setattr(reaper, "_signal_pid", lambda pid: order.append(("kill", pid)))
    monkeypatch.setattr(reaper, "_pid_gone", lambda pid: True)
    monkeypatch.setattr(reaper, "_SETTLE_SEC", 0)
    monkeypatch.setattr(reaper, "_CONFIRM_DEADLINE_SEC", 0)
    monkeypatch.setattr(
        "ouroboros.platform_layer.process_command", lambda pid: OURS,
    )
    monkeypatch.setattr(
        reaper, "pid_environment_assignment_state",
        lambda pid, key, value: containment.ENV_ASSIGNMENT_PRESENT,
    )

    server_paths = reaper.install_server_path_forms(REPO)
    data_dir_values = reaper._path_forms(DATA)
    assert reaper._revalidate_and_kill(9990901, server_paths, data_dir_values) is True
    assert order == [("enum", 9990901), ("kill", 9990901), ("kill", 9990902)]


def test_descendant_discovery_uses_the_shared_platform_seam():
    """The reaper owns identity policy, while generic tree discovery stays in
    platform_layer so its OS behavior cannot drift into a second copy."""
    source = inspect.getsource(reaper)
    assert "_pl.collect_descendant_pids(pid," in source
    assert "def _descendants(" not in source
