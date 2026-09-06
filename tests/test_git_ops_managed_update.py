"""The managed update: its remote, its target and the branch it may not rewrite.

Split verbatim out of ``tests/test_git_ops_recovery.py`` by theme. This module owns the
passive status that does not ensure a remote, the fetch and dependency sync that are panic
tracked and killed on timeout, the manifest remote name the target uses, the dev branch the
preparation preserves, and the pinned checkout a stand keeps across restarts.
"""

from __future__ import annotations

import subprocess


import supervisor.git_ops as git_ops


def test_compute_managed_update_status_passive_does_not_ensure_remote(monkeypatch):
    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {
            "managed_remote_name": "managed",
            "managed_remote_branch": "ouroboros",
        },
    )
    monkeypatch.setattr(
        git_ops,
        "ensure_official_update_remote",
        lambda: (_ for _ in ()).throw(AssertionError("passive status mutated remotes")),
    )
    monkeypatch.setattr(
        git_ops,
        "_resolve_managed_update_target",
        lambda *_args: ("", "", "no cached official tags"),
    )

    def fake_git_capture(cmd):
        if cmd[:3] == ["git", "remote", "get-url"]:
            return 0, "https://github.com/razzant/ouroboros", ""
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "ouroboros", ""
        if cmd == ["git", "rev-parse", "HEAD"]:
            return 0, "abc123", ""
        if cmd == ["git", "status", "--porcelain"]:
            return 0, "", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    status = git_ops.compute_managed_update_status(fetch=False)

    assert status["managed"] is True
    assert "official_status_requires_check" in status["warnings"]

def test_official_fetch_timeout_kills_the_process_tree(monkeypatch):
    import ouroboros.platform_layer as platform_layer
    from ouroboros.tools import shell

    calls = []

    class FakeProcess:
        returncode = 1

        def __init__(self):
            self.communicates = 0

        def communicate(self, input=None, timeout=None):
            assert self in shell._active_subprocesses
            self.communicates += 1
            if self.communicates == 1:
                raise subprocess.TimeoutExpired(["git", "fetch"], timeout)
            return "", "still running"

    proc = FakeProcess()
    monkeypatch.setattr(git_ops.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(
        platform_layer,
        "kill_process_tree",
        lambda child: calls.append(child),
    )

    rc, out, error = git_ops.git_fetch_bounded("managed", timeout=0.01)

    assert rc == git_ops.FETCH_TIMEOUT_RC
    assert out == ""
    assert "exceeded" in error
    assert calls == [proc]
    assert proc not in shell._active_subprocesses

def test_dependency_sync_is_panic_tracked_and_killed_on_timeout(monkeypatch, tmp_path):
    import ouroboros.platform_layer as platform_layer
    from ouroboros.tools import shell

    # The timeout branch LOGS through git_ops.DRIVE_ROOT; unbound, this test is
    # one process-global drift away from appending to the LIVE supervisor log
    # (observed: nondeterministic live writes during full-battery serial runs).
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    (tmp_path / "data" / "logs").mkdir(parents=True)

    killed = []

    class HungProcess:
        returncode = 1

        def __init__(self):
            self.waits = 0

        def wait(self, timeout=None):
            assert self in shell._active_subprocesses
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["pip", "install"], timeout)
            return -9

    proc = HungProcess()
    monkeypatch.setattr(git_ops.subprocess, "Popen", lambda *_a, **_k: proc)
    monkeypatch.setattr(platform_layer, "kill_process_tree", lambda value: killed.append(value))

    ok, _message = git_ops.sync_runtime_dependencies("managed_update_test")

    assert ok is False
    assert killed == [proc]
    assert proc not in shell._active_subprocesses

def test_managed_update_target_uses_manifest_remote_name(monkeypatch):
    import ouroboros.update_channels as update_channels

    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {
            "managed_remote_name": "official",
            "managed_remote_branch": "ouroboros",
        },
    )
    monkeypatch.setattr(update_channels, "get_update_branch", lambda settings=None: "main")

    remote_name, remote_branch, target_ref = git_ops._managed_update_target()

    assert remote_name == "official"
    assert remote_branch == "main"
    assert target_ref == "official/main"

def test_prepare_managed_update_preserves_dev_branch_not_current_head(monkeypatch, tmp_path):
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {"managed_remote_name": "managed"})
    monkeypatch.setattr(git_ops, "_managed_update_target", lambda: ("managed", "main", "managed/main"))
    monkeypatch.setattr(
        git_ops,
        "_resolve_managed_update_target",
        lambda *_args: ("refs/ouroboros-managed/tags/v6.87.5", "remote-sha", ""),
    )
    monkeypatch.setattr(
        git_ops,
        "_collect_repo_sync_state",
        lambda: {"current_branch": "ouroboros", "dirty_lines": [], "unpushed_lines": [], "warnings": []},
    )
    monkeypatch.setattr(
        git_ops,
        "_create_rescue_snapshot",
        lambda **_kwargs: {
            "path": str(tmp_path / "data" / "archive" / "rescue" / "x"),
            "untracked": {"copied_files": 0, "skipped_files": 0, "truncated": False},
        },
    )
    intent_writes = []
    monkeypatch.setattr(git_ops, "_write_update_intent", lambda payload: intent_writes.append(payload))
    monkeypatch.setattr(git_ops, "append_jsonl", lambda _path, _payload: None)

    capture_calls = []

    def fake_git_capture(cmd):
        capture_calls.append(cmd)
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "ouroboros", ""
        if cmd == ["git", "rev-parse", "--verify", "HEAD"]:
            return 0, "base-sha", ""
        if cmd == ["git", "rev-parse", "--verify", "managed/main^{commit}"]:
            return 0, "remote-sha", ""
        if cmd == ["git", "rev-list", "--left-right", "--count", "ouroboros...remote-sha"]:
            return 0, "1 0", ""
        if cmd[:2] == ["git", "branch"] and cmd[-1] == "ouroboros":
            return 0, "", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    ok, payload = git_ops.prepare_managed_update(
        "replace", expected_base_sha="base-sha", expected_target_sha="remote-sha",
        arm_intent=False,
    )

    assert ok is True
    assert payload["keep_branch"].startswith("local-keep-")
    assert payload["update_intent"]["target_sha"] == "remote-sha"
    assert intent_writes == []
    assert any(cmd[:2] == ["git", "branch"] and cmd[-1] == "ouroboros" for cmd in capture_calls)

def test_prepare_managed_update_blocks_when_ahead_check_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {"managed_remote_name": "managed"})
    monkeypatch.setattr(git_ops, "_managed_update_target", lambda: ("managed", "main", "managed/main"))
    monkeypatch.setattr(
        git_ops,
        "_resolve_managed_update_target",
        lambda *_args: ("refs/ouroboros-managed/tags/v6.87.5", "remote-sha", ""),
    )
    monkeypatch.setattr(
        git_ops,
        "_collect_repo_sync_state",
        lambda: {"current_branch": "ouroboros", "dirty_lines": [], "unpushed_lines": [], "warnings": []},
    )
    monkeypatch.setattr(
        git_ops,
        "_create_rescue_snapshot",
        lambda **_kwargs: {
            "path": str(tmp_path / "data" / "archive" / "rescue" / "x"),
            "untracked": {"copied_files": 0, "skipped_files": 0, "truncated": False},
        },
    )

    def fake_git_capture(cmd):
        if cmd[:3] == ["git", "remote", "get-url"]:
            return 0, "https://github.com/razzant/ouroboros", ""
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "ouroboros", ""
        if cmd == ["git", "rev-parse", "--verify", "HEAD"]:
            return 0, "base-sha", ""
        if cmd == ["git", "rev-parse", "--verify", "managed/main^{commit}"]:
            return 0, "remote-sha", ""
        if cmd == ["git", "rev-list", "--left-right", "--count", "ouroboros...remote-sha"]:
            return 128, "", "bad revision"
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    ok, payload = git_ops.prepare_managed_update(
        "replace", expected_base_sha="base-sha", expected_target_sha="remote-sha"
    )

    assert ok is False
    assert "Could not compare local branch with managed update target" in payload["error"]

def test_safe_restart_fallback_does_not_rewrite_dev_branch(monkeypatch):
    checkout_calls = []

    def fake_checkout(branch, reason="unspecified", unsynced_policy="ignore"):
        checkout_calls.append((branch, reason, unsynced_policy))
        return True, "ok"

    import_results = [
        {"ok": False, "stdout": "", "stderr": "broken dev", "returncode": 1},
        {"ok": True, "stdout": "import_ok", "stderr": "", "returncode": 0},
    ]

    monkeypatch.setattr(git_ops, "checkout_and_reset", fake_checkout)
    monkeypatch.setattr(git_ops, "sync_runtime_dependencies", lambda reason: (True, reason))
    monkeypatch.setattr(git_ops, "import_test", lambda: import_results.pop(0))
    monkeypatch.setattr(git_ops, "append_jsonl", lambda _path, _payload: None)

    ok, message = git_ops.safe_restart(reason="owner_restart", unsynced_policy="rescue_and_reset")

    assert ok is True
    assert message == "OK: fell back to ouroboros-stable"
    assert checkout_calls == [
        ("ouroboros", "owner_restart", "rescue_and_reset"),
        ("ouroboros-stable", "owner_restart_fallback_stable", "rescue_and_reset"),
    ]

def test_a_stand_can_keep_its_pinned_checkout_across_restarts(monkeypatch):
    """OUROBOROS_DISABLE_MANAGED_UPDATES=1 is the lever for running a stand.

    A test stand launched against a PINNED checkout had that checkout moved under
    the operator mid-test: the launcher-managed path resets the repo onto the
    managed dev branch on every start (reflog "checkout: moving from <sha> to
    ouroboros", version 6.89.0 -> 6.87.5). server.py already had a local-dev
    branch that skips the BOOTSTRAP reset, but bootstrap is only one of three
    callers — the owner restart and the agent restart reset the tree too. The
    lever therefore sits at `safe_restart`, the choke point all three share, and
    keeps the parts that are not a tree move: deps sync and the import test.
    """
    monkeypatch.setenv("OUROBOROS_DISABLE_MANAGED_UPDATES", "1")

    def fail_checkout(*_args, **_kwargs):
        raise AssertionError("a stand with managed updates disabled must not be checked out")

    events = []
    deps = []
    monkeypatch.setattr(git_ops, "checkout_and_reset", fail_checkout)
    monkeypatch.setattr(git_ops, "sync_runtime_dependencies",
                        lambda reason: deps.append(reason) or (True, reason))
    monkeypatch.setattr(git_ops, "import_test",
                        lambda: {"ok": True, "stdout": "", "stderr": "", "returncode": 0})
    monkeypatch.setattr(git_ops, "append_jsonl", lambda _path, payload: events.append(payload))

    ok, message = git_ops.safe_restart(reason="bootstrap", unsynced_policy="rescue_and_reset")
    assert ok is True
    assert "managed checkout disabled" in message
    assert deps == ["bootstrap"], "the deps sync is not a tree move and must still run"
    assert [e["type"] for e in events] == ["managed_checkout_disabled"], \
        "a suppressed checkout is disclosed, never silent"

    # A broken tree still fails closed — the lever pins the checkout, it does not
    # promise the pinned checkout imports.
    monkeypatch.setattr(git_ops, "import_test",
                        lambda: {"ok": False, "stdout": "", "stderr": "boom", "returncode": 1})
    ok_broken, message_broken = git_ops.safe_restart(reason="owner_restart")
    assert ok_broken is False
    assert "Import test failed" in message_broken

    # Without the lever nothing changes: the ordinary managed path still runs.
    monkeypatch.delenv("OUROBOROS_DISABLE_MANAGED_UPDATES")
    checkouts = []
    monkeypatch.setattr(git_ops, "checkout_and_reset",
                        lambda branch, reason="unspecified", unsynced_policy="ignore":
                        checkouts.append(branch) or (True, "ok"))
    monkeypatch.setattr(git_ops, "import_test",
                        lambda: {"ok": True, "stdout": "", "stderr": "", "returncode": 0})
    assert git_ops.safe_restart(reason="bootstrap")[0] is True
    assert checkouts == [git_ops.BRANCH_DEV]

def test_configure_remote_adds_origin_even_when_managed_remote_exists(monkeypatch):
    calls = []

    monkeypatch.setattr(git_ops, "_has_remote", lambda name=None: name in (None, "managed"))
    monkeypatch.setattr(
        git_ops,
        "git_capture",
        lambda cmd: calls.append(cmd) or (0, "", ""),
    )
    monkeypatch.setattr(
        git_ops,
        "_configure_credential_helper",
        lambda repo_slug, token: calls.append(("helper", repo_slug, token)),
    )

    ok, message = git_ops.configure_remote("razzant/ouroboros", "ghp_test")

    assert ok
    assert message == "ok"
    assert ["git", "remote", "add", "origin", "https://github.com/razzant/ouroboros.git"] in calls

def test_collect_repo_sync_state_prefers_managed_remote(monkeypatch):
    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {
            "managed_remote_name": "managed",
            "managed_remote_branch": "ouroboros",
        },
    )
    def fake_git_capture(cmd, *, timeout=None):
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "ouroboros", ""
        if cmd == ["git", "status", "--porcelain"]:
            return 0, "", ""
        if cmd == ["git", "remote"]:
            return 0, "managed", ""
        if cmd == ["git", "log", "--oneline", "managed/ouroboros..HEAD"]:
            return 0, "abc123 local commit\n", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    state = git_ops._collect_repo_sync_state()

    assert state["current_branch"] == "ouroboros"
    assert state["unpushed_lines"] == ["abc123 local commit"]

def test_checkout_and_reset_keeps_bundled_sha_on_first_managed_bootstrap(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / git_ops.BOOTSTRAP_PIN_MARKER_NAME).write_text("pending\n", encoding="utf-8")

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "_has_remote", lambda name=None: name in (None, "managed"))
    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {
            "managed_remote_name": "managed",
            "managed_remote_branch": "ouroboros",
            "source_sha": "bundle123",
        },
    )
    monkeypatch.setattr(git_ops, "load_state", lambda: {"current_sha": "bundle123"})

    saved_state = {}
    monkeypatch.setattr(git_ops, "save_state", lambda state: saved_state.update(state))

    def fake_git_capture(cmd):
        if cmd == ["git", "rev-parse", "HEAD"]:
            return 0, "bundle123", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    calls = []

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        calls.append(cmd)
        if cmd == ["git", "rev-parse", "--verify", "ouroboros"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="bundle123\n", stderr="")
        if cmd[:2] == ["git", "checkout"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "reset"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"] and cmd[-1] == "HEAD":
            return subprocess.CompletedProcess(cmd, 0, stdout="bundle123\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset("ouroboros", reason="bootstrap", unsynced_policy="ignore")

    assert ok
    assert message == "ok"
    assert ["git", "fetch", "managed"] not in calls
    assert saved_state["current_sha"] == "bundle123"
    assert not (git_dir / git_ops.BOOTSTRAP_PIN_MARKER_NAME).exists()

def test_ensure_official_update_remote_uses_manifest_remote_name(monkeypatch):
    captured = []
    monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {"managed_remote_name": "official"})
    monkeypatch.setattr(git_ops, "_list_remotes", lambda: [])
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd: captured.append(cmd) or (0, "", ""))
    ok, _msg = git_ops.ensure_official_update_remote()
    assert ok
    assert ["git", "remote", "add", "official", git_ops.OFFICIAL_UPDATE_REMOTE_URL] in captured


def test_ensure_official_update_remote_honors_configured_source_across_cycles(monkeypatch):
    """W2-F2 (owner №4=A): a configured ``managed_remote_url`` is honored on
    EVERY update fetch — N repin cycles never retarget a fork/mirror/air-gap
    install to the hardcoded official URL."""
    configured = "https://mirror.example.invalid/fork/ouroboros.git"
    captured = []
    monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {
        "managed_remote_name": "managed",
        "managed_remote_url": configured,
    })
    monkeypatch.setattr(git_ops, "_list_remotes", lambda: ["managed"])
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd: captured.append(cmd) or (0, "", ""))
    for _ in range(3):
        ok, _msg = git_ops.ensure_official_update_remote()
        assert ok
    assert captured == [["git", "remote", "set-url", "managed", configured]] * 3
    assert not any(git_ops.OFFICIAL_UPDATE_REMOTE_URL in cmd for cmd in captured)


def test_ensure_official_update_remote_blank_configured_source_defaults(monkeypatch):
    """No configured source (or a blank one) keeps the former default: the
    hardcoded official repository URL."""
    captured = []
    monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {"managed_remote_url": "   "})
    monkeypatch.setattr(git_ops, "_list_remotes", lambda: ["managed"])
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd: captured.append(cmd) or (0, "", ""))
    ok, _msg = git_ops.ensure_official_update_remote()
    assert ok
    assert captured == [
        ["git", "remote", "set-url", "managed", git_ops.OFFICIAL_UPDATE_REMOTE_URL]
    ]
    assert git_ops.managed_update_remote_url({"managed_remote_url": "   "}) == (
        git_ops.OFFICIAL_UPDATE_REMOTE_URL)
    assert git_ops.managed_update_remote_url({}) == git_ops.OFFICIAL_UPDATE_REMOTE_URL
