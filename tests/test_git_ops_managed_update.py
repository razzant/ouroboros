"""The managed update: its remote, its target and the branch it may not rewrite.

Split verbatim out of ``tests/test_git_ops_recovery.py`` by theme. This module owns the
passive status that does not ensure a remote, the fetch and dependency sync that are panic
tracked and killed on timeout, the manifest remote name the target uses, the dev branch the
preparation preserves, and the pinned checkout a stand keeps across restarts.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

import supervisor.git_ops as git_ops
from tests._git_ops_recovery_shared import _git, _history_repo


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
        if cmd == ["git", "--no-optional-locks", "status", "--porcelain"]:
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
        def poll(self):
            return self.returncode

        returncode = 1
        stdin = stdout = stderr = None  # the runner closes an exited child's pipes; this fake opens none

        def __init__(self):
            self.communicates = 0

        def communicate(self, input=None, timeout=None):
            assert self in shell._active_subprocesses
            self.communicates += 1
            if self.communicates == 1:
                raise subprocess.TimeoutExpired(["git", "fetch"], timeout)
            return "", "still running"

    proc = FakeProcess()
    from tests.test_tracked_tree_containment import fake_tree_read
    fake_tree_read(monkeypatch, proc, remains=False)
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
    # This case deliberately exercises the PRODUCTION install path, so it drops
    # the test-boundary marker the suite sets (see the boundary test below).
    monkeypatch.delenv("OUROBOROS_PYTEST_ACTIVE", raising=False)

    killed = []

    class HungProcess:
        def poll(self):
            return self.returncode

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
    from tests.test_tracked_tree_containment import fake_tree_read
    fake_tree_read(monkeypatch, proc, remains=False)
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

    monkeypatch.setattr(git_ops, "_is_launcher_managed_repo", lambda: True)
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
    monkeypatch.setattr(git_ops, "_is_launcher_managed_repo", lambda: True)
    checkouts = []
    monkeypatch.setattr(git_ops, "checkout_and_reset",
                        lambda branch, reason="unspecified", unsynced_policy="ignore":
                        checkouts.append(branch) or (True, "ok"))
    monkeypatch.setattr(git_ops, "import_test",
                        lambda: {"ok": True, "stdout": "", "stderr": "", "returncode": 0})
    assert git_ops.safe_restart(reason="bootstrap")[0] is True
    assert checkouts == [git_ops.BRANCH_DEV]


@pytest.fixture
def source_checkout(tmp_path, monkeypatch):
    """A team's own checkout: its own branch, unsaved edits, and an origin nobody asked to fetch."""
    from supervisor import git_ops_reset

    repo, first, second = _history_repo(tmp_path)
    _git(repo, "branch", "ouroboros-stable", first)
    _git(repo, "checkout", "-q", "-b", "team-feature")
    (repo / "value.txt").write_text("unsaved edit\n", encoding="utf-8")
    (repo / "notes.txt").write_text("untracked notes\n", encoding="utf-8")
    _git(repo, "remote", "add", "origin", str(tmp_path / "origin-nobody-asked-for"))
    drive = tmp_path / "data"
    (drive / "logs").mkdir(parents=True)
    for name, value in {"REPO_DIR": repo, "DRIVE_ROOT": drive, "BRANCH_DEV": "ouroboros",
                        "BRANCH_STABLE": "ouroboros-stable"}.items():
        monkeypatch.setattr(git_ops, name, value)
    monkeypatch.setattr(git_ops, "update_state", lambda mutate, **_kw: mutate({}))
    monkeypatch.setattr(git_ops, "append_jsonl", lambda _path, row: events.append(row))
    for name in ("OUROBOROS_DISABLE_MANAGED_UPDATES", "OUROBOROS_MANAGED_BY_LAUNCHER", "OUROBOROS_MANAGED_REPO_DIR"):
        monkeypatch.delenv(name, raising=False)
    events, deps = [], []
    monkeypatch.setattr(git_ops, "sync_runtime_dependencies", lambda reason: deps.append(reason) or (True, "unchanged"))
    monkeypatch.setattr(git_ops, "import_test", lambda: {"ok": True, "stdout": "", "stderr": "", "returncode": 0})

    def identity():
        return _git(repo, "symbolic-ref", "--short", "HEAD"), _git(repo, "rev-parse", "HEAD")

    def served():
        return (*identity(), _git(repo, "status", "--porcelain"), (repo / "value.txt").read_bytes(),
                (repo / "notes.txt").read_bytes(), (repo / ".git" / "FETCH_HEAD").exists())

    def no_tree_move(*_a, **_k):
        raise AssertionError("this restart has no authorized tree move")

    def forbid_moves():
        monkeypatch.setattr(git_ops, "checkout_and_reset", no_tree_move)
        monkeypatch.setattr(git_ops_reset, "_recovery_checkout", no_tree_move)

    return SimpleNamespace(repo=repo, first=first, second=second, identity=identity, served=served,
                           forbid_moves=forbid_moves, deps=deps, events=events)


@pytest.mark.parametrize(("reason", "policy"), [("owner_restart", "rescue_and_reset"),
                                                ("agent_restart_request", "rescue_and_block")])
def test_restart_on_a_source_checkout_serves_what_is_on_disk(source_checkout, reason, policy):
    scene = source_checkout
    scene.forbid_moves()
    before = scene.served()

    ok, message = git_ops.safe_restart(reason=reason, unsynced_policy=policy)

    assert ok is True and message == f"OK: unmanaged checkout kept at {before[1][:12]}"
    assert scene.served() == before and before[0] == "team-feature" and before[5] is False
    assert scene.deps == [reason]
    assert [(row["type"], row["head"]) for row in scene.events] == [("unmanaged_checkout_kept", before[1])]


@pytest.mark.parametrize("failing", ["deps", "import"])
def test_a_source_checkout_refuses_a_broken_restart_and_changes_nothing(source_checkout, monkeypatch, failing):
    scene = source_checkout
    scene.forbid_moves()
    if failing == "deps":
        monkeypatch.setattr(git_ops, "sync_runtime_dependencies", lambda reason: (False, "pip exited 1"))
        expected = "Failed deps on this unmanaged checkout: pip exited 1; the checkout was left untouched."
    else:
        monkeypatch.setattr(git_ops, "import_test", lambda: {
            "ok": False, "stdout": "", "stderr": "Traceback\nModuleNotFoundError: pypdf\n", "returncode": 1})
        expected = ("Import test failed on this unmanaged checkout (rc=1): ModuleNotFoundError: pypdf; "
                    "the checkout was left untouched.")
    before = scene.served()

    assert git_ops.safe_restart(reason="owner_restart") == (False, expected)
    assert scene.served() == before


@pytest.mark.parametrize("marker", ["launcher_names_this_repo", "metadata_names_this_repo"])
def test_a_launcher_managed_checkout_still_returns_to_its_working_branch(source_checkout, monkeypatch, marker):
    scene = source_checkout
    monkeypatch.setattr(git_ops, "_has_remote", lambda *_a: False)
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    if marker == "launcher_names_this_repo":
        monkeypatch.setenv("OUROBOROS_MANAGED_REPO_DIR", str(scene.repo))
    else:
        (scene.repo / ".git" / git_ops.MANAGED_REPO_META_NAME).write_text(
            json.dumps({"managed_repo_dir": str(scene.repo.resolve())}), encoding="utf-8")

    assert git_ops.safe_restart(reason="owner_restart") == (True, "OK: ouroboros")
    assert scene.identity() == ("ouroboros", scene.second) and scene.deps == ["owner_restart"]
    assert "unmanaged_checkout_kept" not in [row["type"] for row in scene.events]


@pytest.mark.parametrize("identity", ["marker_for_another_repo", "metadata_without_marker",
                                      "stray_intent_without_metadata", "intent_for_another_branch"])
def test_nothing_short_of_boot_identity_or_an_armed_update_moves_a_source_checkout(source_checkout, monkeypatch,
                                                                                   identity):
    scene = source_checkout
    scene.forbid_moves()
    metadata = json.dumps({"managed_repo_dir": str(scene.repo.resolve())})
    if identity == "marker_for_another_repo":
        monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
        monkeypatch.setenv("OUROBOROS_MANAGED_REPO_DIR", str(scene.repo.parent / "another-repo"))
    if identity in {"metadata_without_marker", "intent_for_another_branch"}:
        (scene.repo / ".git" / git_ops.MANAGED_REPO_META_NAME).write_text(metadata, encoding="utf-8")
    if identity == "stray_intent_without_metadata":
        git_ops._write_update_intent({"branch": "ouroboros", "target_sha": scene.first})
    if identity == "intent_for_another_branch":
        git_ops._write_update_intent({"branch": "another", "target_sha": scene.first})
    before = scene.served()

    assert git_ops.safe_restart(reason="owner_restart")[0] is True
    assert scene.served() == before


def test_an_armed_update_still_lands_through_restart_on_an_unlaunched_checkout(source_checkout, monkeypatch):
    """Update's replay: the intent `prepare_managed_update` armed is applied by the restart that follows."""
    from supervisor import git_ops_reset, update_merge

    scene = source_checkout
    monkeypatch.setattr(git_ops, "_has_remote", lambda *_a: False)
    target = _git(scene.repo, "commit-tree", "ouroboros^{tree}", "-p", "ouroboros", "-m", "official")
    (scene.repo / ".git" / git_ops.MANAGED_REPO_META_NAME).write_text(
        json.dumps({"managed_repo_dir": str(scene.repo.resolve())}), encoding="utf-8")
    git_ops._write_update_intent({"branch": "ouroboros", "target_sha": target})
    monkeypatch.setattr(update_merge, "read_update_tx_strict", lambda: (
        "valid", {"phase": "applying_replace", "target_sha": target, "pre_update_branch": "ouroboros"}))
    monkeypatch.setattr(git_ops._update_source, "official_ref_has_constitution", lambda *_a, **_k: True)
    monkeypatch.setattr(git_ops_reset, "_target_lacks_store", lambda _sha: False)

    assert git_ops.safe_restart(reason="owner_restart") == (True, "OK: ouroboros")
    assert scene.identity() == ("ouroboros", target) and not git_ops._read_update_intent()


def test_restore_is_not_undone_by_the_restart_that_follows_it(source_checkout):
    """Restore moves the tree itself; the restart after it must serve exactly that."""
    scene = source_checkout
    ok, message = git_ops.rollback_to_version(scene.first)
    assert ok is True, message
    scene.forbid_moves()

    assert git_ops.safe_restart(reason="owner_restart")[0] is True
    assert scene.identity() == ("team-feature", scene.first)


@pytest.mark.parametrize("keeper", ["managed_checkout_disabled", "unmanaged_checkout_kept", "managed_by_deployment"])
def test_a_kept_checkout_drops_a_stale_stamped_identity_and_nothing_else(source_checkout, monkeypatch, keeper):
    """A branch/SHA an earlier tree move stamped would outrank the live read of a kept checkout, so the keep clears
    it without stamping an observed SHA or forgetting the smoke receipt; the deployment's own projection stays."""
    from ouroboros import config
    from ouroboros.gateway.state import _runtime_repo_identity
    from supervisor import git_ops_reset

    scene = source_checkout
    scene.forbid_moves()
    stale = {"current_branch": "ouroboros", "current_sha": scene.first, "owner_chat_id": 7}
    state = dict(stale)
    monkeypatch.setattr(git_ops, "update_state", lambda mutate, **_kw: mutate(state))
    monkeypatch.setattr(config, "REPO_DIR", scene.repo)
    if keeper == "managed_checkout_disabled":
        monkeypatch.setenv("OUROBOROS_DISABLE_MANAGED_UPDATES", "1")
    if keeper == "managed_by_deployment":
        monkeypatch.setattr(git_ops_reset, "body_replacement_refusal", lambda: "the administrator's program")
    receipt, watermarks = git_ops_reset._bootstrap_receipt_path(), git_ops.DRIVE_ROOT / "state" / "migrations.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"schema_version": 1, "head": scene.first}), encoding="utf-8")
    watermarks.write_text(json.dumps({"schema_generation": 7}), encoding="utf-8")
    on_disk = receipt.read_bytes(), watermarks.read_bytes()

    assert git_ops.safe_restart(reason="owner_restart")[0] is True

    assert [row["type"] for row in scene.events] == [keeper]
    assert (receipt.read_bytes(), watermarks.read_bytes()) == on_disk
    if keeper == "managed_by_deployment":
        assert state == stale
    else:
        assert state == {**stale, "current_branch": None, "current_sha": None}
        assert _runtime_repo_identity(state) == ("team-feature", scene.second)


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
    monkeypatch.setattr(git_ops, "update_state", lambda mutator, **_kw: mutator(saved_state) or saved_state)

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


@pytest.mark.parametrize("legacy", [
    "https://github.com/joi-lab/ouroboros-desktop",  # what builds before v6.8.0 store
    "git@github.com:joi-lab/ouroboros.git",
    "HTTPS://GitHub.com/Joi-Lab/Ouroboros-Desktop/",
])
def test_former_official_home_updates_from_the_official_repository(monkeypatch, legacy):
    """Owner 2026-10-11 №3=A: an archived former official home is the project's
    own retired address, so every repin targets the official repository."""
    captured = []
    monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {
        "managed_remote_name": "managed",
        "managed_remote_url": legacy,
    })
    monkeypatch.setattr(git_ops, "_list_remotes", lambda: ["managed"])
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd: captured.append(cmd) or (0, "", ""))
    for _ in range(2):
        ok, _msg = git_ops.ensure_official_update_remote()
        assert ok
    assert captured == [["git", "remote", "set-url", "managed", git_ops.OFFICIAL_UPDATE_REMOTE_URL]] * 2


@pytest.mark.parametrize("configured", [
    "https://github.com/joi-lab/ouroboros-joi",  # another repository of the same organisation
    "joi-lab/ouroboros",  # not a GitHub URL
    "/srv/mirrors/joi-lab/ouroboros.git",
])
def test_only_the_exact_former_homes_are_mapped(configured):
    """The W2-F2 rule stays intact for every other configured source."""
    assert git_ops.managed_update_remote_url({"managed_remote_url": configured}) == configured


def test_dependency_sync_never_installs_into_an_interpreter_under_test(monkeypatch, tmp_path):
    """The ONE chokepoint refuses installs for every caller, not just bootstrap."""
    import server

    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    (tmp_path / "data" / "logs").mkdir(parents=True)
    spawned = []
    monkeypatch.setattr(git_ops.subprocess, "Popen",
                        lambda *a, **k: spawned.append(a) or _NeverRun())
    monkeypatch.setenv("OUROBOROS_PYTEST_ACTIVE", "1")

    assert git_ops.sync_runtime_dependencies("bootstrap_local_dev") == (True, "pytest:suppressed")
    assert git_ops.sync_runtime_dependencies("managed_update_pre_restart") == (True, "pytest:suppressed")
    assert spawned == [], "a test-boundary run reached the pip call site"

    # The server bootstrap keeps calling the shared chokepoint: the suppression is
    # a property of the dependency sync, never of one caller's local-dev branch.
    calls = []
    monkeypatch.setattr(server, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", False)
    monkeypatch.setattr(server, "setup_remote_if_configured", lambda *args: None)

    class _Module:
        def init(self, **_kwargs):
            return None

        def ensure_repo_present(self):
            return None

        def sync_runtime_dependencies(self, reason):
            calls.append(reason)
            return git_ops.sync_runtime_dependencies(reason)

        def import_test(self):
            return {"ok": True}

    assert server._bootstrap_supervisor_repo({}, _Module())[0]
    assert calls == ["bootstrap_local_dev"]
    assert spawned == []


class _NeverRun:
    def wait(self, timeout=None):  # pragma: no cover - the boundary must precede this
        raise AssertionError("the suppressed path must not wait on a pip process")
