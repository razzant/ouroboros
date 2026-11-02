"""#1802: registered disposition returns only to the original target under current rights.

All owners, captures, engine responses and working trees here are synthetic. The
engine fixture owns directory CAS; Git applies exercise the real host materializer.
"""
from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from ouroboros import delegate_custody as custody, delegate_custody_current as current
from ouroboros.delegate_directory import capture_directory_result
from ouroboros.subagent_worktrees import provision_execution_snapshot, provision_genesis_project, provision_worktree
from ouroboros.task_results import write_task_result
from ouroboros.tools.delegate import _capture_terminal_patch
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_delegated_directory import DirectoryEngine
from tests.test_delegated_run_isolation import _git
from tests.test_isolated_project_copies import repo


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    installation = tmp_path / "installation"
    installation.mkdir()
    data, projects, copies = (installation / name for name in ("data", "projects", "copies"))
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(copies))
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    monkeypatch.delenv("OUROBOROS_BOOT_RUNTIME_MODE", raising=False)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    system = repo(tmp_path / "system", {"BIBLE.md": b"body\n", "content.txt": b"base\n"})
    active = repo(tmp_path / "active", {"untouched.txt": b"active input\n"})
    ctx = ToolContext(repo_dir=system, drive_root=data, task_id="recovery",
                      workspace_root=str(active), workspace_mode="external", task_metadata={})
    custody._CUSTODY.clear()
    yield ctx, projects, copies
    custody._CUSTODY.clear()


class CasDirectoryEngine(DirectoryEngine):
    """The fake engine implements the existing per-file preimage refusal too."""

    def __init__(self, target, path="content.txt", *, lost_apply=False):
        super().__init__(target, lost_apply=lost_apply)
        self.path = path
        self.before = (target / path).read_bytes() if (target / path).exists() else None
        self.manifest["entries"][0]["path"] = path
        self.raw = json.dumps(self.manifest).encode()

    def apply_run(self, rid, request, *, idempotency_key):
        assert request["target"] == {"kind": "original_project"}
        destination = self.target / self.path
        if (destination.read_bytes() if destination.exists() else None) != self.before:
            return {"applied": False, "refused": True, "reason": "baseline_changed"}
        self.applies.append(idempotency_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.body)
        if self.lost_apply:
            self.lost_apply = False
            raise OSError("response lost after application")
        return {"applied": True, "refused": False, "appliedPaths": [self.path], "treeMutated": True}


def capture(recovery, monkeypatch, *, kind="git", area="projects", terminal=True, prepared=True,
            path="content.txt", body=False, lost_apply=False):
    ctx, projects, copies = recovery
    if area == "projects":
        target = Path(provision_genesis_project(repo_dir=ctx.repo_dir, task_id="nanny",
                                               projects_root=projects, data_dir=ctx.drive_root).path)
        (target / "content.txt").write_bytes(b"base\n")
        _git(target, "add", "content.txt")
        _git(target, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
             "commit", "-qm", "base")
    elif area == "copies":
        source = ctx.repo_dir if body else Path(ctx.workspace_root)
        target = Path(provision_worktree(repo_dir=source, task_id="nanny", worktree_root=copies,
                                         data_dir=ctx.drive_root, source_is_system_repo=body).path)
    else:
        target = repo(projects.parent.parent / "attached", {"content.txt": b"base\n"})
    engine = None
    if kind == "directory":
        target = target / "documents"
        target.mkdir()
        (target / "content.txt").write_bytes(b"base\n")
        engine = CasDirectoryEngine(target, path, lost_apply=lost_apply)
        entry = custody.RunCustody(run_id="run", task_id="nanny", route_id="fixture",
            target_root=str(target), authority_source="external_workspace_root", access="full",
            resource_ref={"workspace_kind": "directory", "strategy": "copy", "scopePaths": ["."]})

        @contextmanager
        def gateway():
            yield engine

        monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", gateway)
    else:
        handle = provision_execution_snapshot(target_root=target, task_id="nanny", snapshot_id="result",
                                              worktree_root=copies, data_dir=ctx.drive_root)
        output = Path(handle.path) / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"accepted child result\n")
        entry = custody.RunCustody(run_id="run", task_id="nanny", route_id="fixture",
            target_root=str(target), authority_source="external_workspace_root", access="full",
            snapshot_id=handle.snapshot_id, execution_root=handle.path, baseline_sha=handle.baseline_sha)
    assert custody.record_started(ctx.drive_root, entry)
    assert custody.emit(ctx.drive_root, custody.SETTLED, {"run_id": "run", "task_id": "nanny"})
    entry = current.run(ctx.drive_root, "run")
    if terminal:
        write_task_result(ctx.drive_root, "nanny", "failed")
    if prepared:
        block = (capture_directory_result(ctx.drive_root, entry, engine) if engine else
                 _capture_terminal_patch(ctx, entry))
        assert block["status"] in {"ready", "ready_with_changes"}, block
    custody._CUSTODY.clear()
    return target, engine


def apply_result(ctx, **kwargs):
    """Invoke the actual registered consumer, not a helper standing in for it."""
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry.execute_result("integrate_delegated_patch", {"run_id": "run", **kwargs})


def apply(ctx, **kwargs):
    return apply_result(ctx, **kwargs).text


def active_state(ctx):
    active = Path(ctx.workspace_root)
    return (_git(active, "status", "--porcelain").stdout,
            _git(active, "rev-parse", "HEAD").stdout, (active / ".git/index").read_bytes(),
            (active / "untouched.txt").read_bytes())


@pytest.mark.parametrize("kind", ["git", "directory"])
@pytest.mark.parametrize("area", ["projects", "copies"])
def test_registered_orphan_returns_to_original_host_target_without_rebinding(recovery, monkeypatch, kind, area):
    ctx = recovery[0]
    target, engine = capture(recovery, monkeypatch, kind=kind, area=area)
    before, focus = active_state(ctx), ctx.workspace_root
    head = _git(target, "rev-parse", "HEAD").stdout if kind == "git" else None
    out = apply(ctx)
    assert ("✅ Integrated" in out if kind == "git" else json.loads(out)["status"] == "applied"), out
    # The folder result is JSON: its target is a decoded field (a Windows path's backslashes are escaped there).
    assert (str(target) in out) if kind == "git" else json.loads(out)["target"] == str(target), out
    expected = engine.body if engine else b"accepted child result\n"
    assert (target / "content.txt").read_bytes() == expected
    if kind == "git":
        assert _git(target, "diff", "--cached", "--name-only").stdout == "content.txt\n"
        assert _git(target, "rev-parse", "HEAD").stdout == head
    else:
        assert len(engine.applies) == 1 and not (target / ".git").exists()
    assert ctx.workspace_root == focus and active_state(ctx) == before
    held = current.run(ctx.drive_root, "run")
    assert held.patch_disposed == "applied"
    disposed = [row for row in custody.custody_rows(ctx.drive_root)
                if row.get("type") == custody.PATCH_DISPOSED and row.get("run_id") == "run"]
    assert [row["disposed_by_task_id"] for row in disposed] == ["recovery"]
    # Durable replay does not repeat a materialization, even if the in-process memo is gone.
    custody._CUSTODY.clear()
    again = apply(ctx)
    assert "INTEGRATE_DELEGATED_NOT_OWNED" in again, again
    assert (target / "content.txt").read_bytes() == expected
    assert len(engine.applies) == 1 if engine else _git(target, "rev-parse", "HEAD").stdout == head
    assert active_state(ctx) == before


def restrict_presence(ctx):
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.frozen_presence_authority import build_presence_capability_ceiling
    from tests.frozen_presence_capabilities import PresenceResourceTarget, PresenceToolTarget
    from tests.test_presence_authority import _resolution

    ceiling = build_presence_capability_ceiling(skill_name="fixture", skill_content_hash="c" * 64,
        state_fingerprint="d" * 64, resolution=_resolution(
            PresenceToolTarget("builtin", "integrate_delegated_patch"),
            PresenceResourceTarget("active_workspace", ("write",), ".")))
    ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}


@pytest.mark.parametrize("kind", ["git", "directory"])
@pytest.mark.parametrize("restriction", ["write", "presence", "readonly", "task_path", "live_owner", "attached", "drift"])
def test_registered_orphan_keeps_current_authority_and_original_preimage(recovery, monkeypatch, kind, restriction):
    ctx = recovery[0]
    target, engine = capture(recovery, monkeypatch, kind=kind,
        area="attached" if restriction == "attached" else "projects", terminal=restriction != "live_owner")
    before = active_state(ctx)
    if restriction == "write":
        monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    elif restriction == "presence":
        restrict_presence(ctx)
    elif restriction in {"readonly", "task_path"}:
        ctx.task_constraint = ({"mode": "local_readonly_subagent"} if restriction == "readonly" else
            {"mode": "acting_subagent", "surface": "external_workspace", "write_root": ctx.workspace_root})
    elif restriction == "drift":
        (target / "content.txt").write_bytes(b"parent moved\n")
    preimage = (target / "content.txt").read_bytes()
    out = apply(ctx)
    assert "✅ Integrated" not in out and not out.startswith('{"status": "applied"'), out
    assert (target / "content.txt").read_bytes() == preimage
    assert active_state(ctx) == before
    assert not current.run(ctx.drive_root, "run").patch_disposed
    assert not current.run(ctx.drive_root, "run").patch_apply_pending
    if engine:
        assert engine.applies == []
    elif restriction != "drift":
        assert _git(target, "diff", "--cached", "--name-only").stdout == ""


@pytest.mark.parametrize("kind", ["git", "directory"])
@pytest.mark.parametrize("actor", ["own", "retry"])
def test_own_and_confirmed_retry_keep_exact_target_equality(recovery, monkeypatch, kind, actor):
    ctx = recovery[0]
    target, engine = capture(recovery, monkeypatch, kind=kind)
    if actor == "own":
        ctx.task_id = "nanny"
    else:
        from ouroboros.delegate_collection import observe_run
        from tests.test_delegate_body_collection import terminal

        assert observe_run(ctx.drive_root, current.run(ctx.drive_root, "run"), terminal("run"))
        monkeypatch.setattr("ouroboros.delegate_shared._confirmed_retry_chain", lambda *a, **k: ("nanny", "recovery"))
    out = apply(ctx)
    assert "INTEGRATE_DELEGATED_TARGET_MISMATCH" in out, out
    assert (target / "content.txt").read_bytes() == b"base\n"
    assert not current.run(ctx.drive_root, "run").patch_disposed
    assert not engine or not engine.applies


@pytest.mark.parametrize("mode,path", [("advanced", "BIBLE.md"), ("light", "content.txt")])
def test_body_policy_uses_recorded_checkout_not_foreign_active_folder(recovery, monkeypatch, mode, path):
    ctx = recovery[0]
    target, _ = capture(recovery, monkeypatch, area="copies", path=path, body=True)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    # It is existing explicit system-repo authority that reaches this linked body checkout.
    ctx.system_repo_dir = target
    out = apply(ctx)
    assert ("CORE_PROTECTION_BLOCKED" if mode == "advanced" else "LIGHT_MODE_BLOCKED") in out, out
    assert (target / path).read_bytes() == (b"body\n" if path == "BIBLE.md" else b"base\n")
    assert not current.run(ctx.drive_root, "run").patch_disposed


@pytest.mark.parametrize("kind", ["git", "directory"])
def test_new_capture_is_returned_before_cross_target_apply(recovery, monkeypatch, kind):
    ctx = recovery[0]
    target, engine = capture(recovery, monkeypatch, kind=kind, prepared=False)
    shown = apply(ctx)
    assert "INTEGRATE_DELEGATED_INSPECT_CAPTURE" in shown, shown
    assert (target / "content.txt").read_bytes() == b"base\n"
    assert not current.run(ctx.drive_root, "run").patch_disposed
    assert not engine or not engine.applies
    out = apply(ctx)
    assert ("✅ Integrated" in out if kind == "git" else json.loads(out)["status"] == "applied"), out


def test_git_rechecks_target_rights_inside_existing_apply_lock(recovery, monkeypatch):
    from ouroboros.tools import git as git_tools

    ctx = recovery[0]
    target, _ = capture(recovery, monkeypatch)
    original = git_tools._acquire_git_lock

    def acquire(*args, **kwargs):
        lock = original(*args, **kwargs)
        restrict_presence(ctx)
        return lock

    monkeypatch.setattr(git_tools, "_acquire_git_lock", acquire)
    result = apply_result(ctx)
    assert result.status == "blocked" and result.code == "INTEGRATION_BLOCKED", result
    out = result.text
    assert "PRESENCE_RESOURCE_BLOCKED" in out, out
    assert "current write authority" in out and str(target) in out
    assert out.count("INTEGRATE_TARGET_FORBIDDEN:") == 1
    assert (target / "content.txt").read_bytes() == b"base\n"
    assert not current.run(ctx.drive_root, "run").patch_apply_pending


def test_directory_rechecks_target_rights_before_engine_submission(recovery, monkeypatch):
    ctx = recovery[0]
    target, engine = capture(recovery, monkeypatch, kind="directory")
    original = custody.record_patch_apply_started

    def intent(*args, **kwargs):
        recorded = original(*args, **kwargs)
        restrict_presence(ctx)
        return recorded

    monkeypatch.setattr(custody, "record_patch_apply_started", intent)
    result = apply_result(ctx)
    assert result.status == "blocked" and result.code == "INTEGRATION_BLOCKED", result
    out = result.text
    assert "PRESENCE_RESOURCE_BLOCKED" in out, out
    assert "current write authority" in out and str(target) in out
    assert out.count("INTEGRATE_TARGET_FORBIDDEN:") == 1
    assert (target / "content.txt").read_bytes() == b"base\n" and engine.applies == []
    assert not current.run(ctx.drive_root, "run").patch_apply_pending


def test_directory_lost_reply_keeps_unknown_intent_not_a_false_disposition(recovery, monkeypatch):
    ctx = recovery[0]
    target, engine = capture(recovery, monkeypatch, kind="directory", lost_apply=True)
    out = apply(ctx)
    assert "INTEGRATE_DELEGATED_APPLY_UNCONFIRMED" in out, out
    assert (target / "content.txt").read_bytes() == engine.body
    assert current.run(ctx.drive_root, "run").patch_apply_pending
    assert not current.run(ctx.drive_root, "run").patch_disposed
    assert "APPLY_AMBIGUOUS" in apply(ctx)
    assert len(engine.applies) == 1


@pytest.mark.parametrize("kind", ["git", "directory"])
def test_late_ownership_refusal_keeps_its_reason_not_write_rights(recovery, monkeypatch, kind):
    from ouroboros.tools import git as git_tools
    from ouroboros.tools import subagent_integration_delegated as integration

    ctx = recovery[0]
    target, engine = capture(recovery, monkeypatch, kind=kind)
    refusal = "⚠️ INTEGRATE_DELEGATED_NOT_OWNED: run ownership could not be read"
    module = git_tools if kind == "git" else custody
    name = "_acquire_git_lock" if kind == "git" else "record_patch_apply_started"
    original = getattr(module, name)

    def change_ownership(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(integration, "_current_disposition_refusal", lambda *_: refusal)
        return result

    monkeypatch.setattr(module, name, change_ownership)
    result = apply_result(ctx)
    assert result.status == "blocked" and result.code == "INTEGRATION_BLOCKED"
    assert result.text == refusal
    assert (target / "content.txt").read_bytes() == b"base\n"
    assert not engine or not engine.applies
    assert not current.run(ctx.drive_root, "run").patch_apply_pending


def test_child_checkout_never_becomes_a_coop_project(recovery, monkeypatch):
    from ouroboros.tools.subagent_integration import _is_host_minted_projects_tree

    target, _ = capture(recovery, monkeypatch, area="copies")
    assert not _is_host_minted_projects_tree(target)
    assert _is_host_minted_projects_tree(recovery[1] / "a-project")


@pytest.mark.parametrize("kind", ["git", "directory"])
def test_preintent_write_refusal_is_typed_once_and_names_the_original_target(recovery, monkeypatch, kind):
    ctx = recovery[0]
    target, engine = capture(recovery, monkeypatch, kind=kind)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    result = apply_result(ctx)
    assert result.status == "blocked" and result.code == "INTEGRATION_BLOCKED", result
    assert result.text.count("INTEGRATE_TARGET_FORBIDDEN:") == 1, result.text
    assert "current write authority" in result.text and str(target) in result.text
    assert not current.run(ctx.drive_root, "run").patch_apply_pending
    assert not engine or not engine.applies


def test_directory_apply_uses_verified_capture_name_after_data_relocation(recovery, monkeypatch):
    ctx = recovery[0]
    target, engine = capture(recovery, monkeypatch, kind="directory")
    cap = custody.delegated_capture_dir(ctx.drive_root, "nanny", "run")
    path = cap / "workspace_patch.json"
    manifest = json.loads(path.read_text())
    for row in [manifest["engine_manifest"], *manifest["file_outputs"]]:
        row["path"] = str(cap.parent / "gone-old-root" / row["name"])
    path.write_text(json.dumps(manifest))
    result = apply_result(ctx)
    assert result.status == "ok", result.text
    assert json.loads(result.text)["status"] == "applied"
    assert (target / "content.txt").read_bytes() == engine.body and len(engine.applies) == 1


def test_unreadable_directory_capture_is_not_an_unconfirmed_apply(recovery, monkeypatch):
    ctx = recovery[0]
    target, engine = capture(recovery, monkeypatch, kind="directory")
    cap = custody.delegated_capture_dir(ctx.drive_root, "nanny", "run")
    (cap / "engine-files-manifest.json").unlink()
    result = apply_result(ctx)
    assert result.status == "blocked" and result.code == "INTEGRATION_BLOCKED", result
    assert "INTEGRATE_MANIFEST_UNREADABLE" in result.text
    assert "UNCONFIRMED" not in result.text and "submitted" in result.text
    assert not engine.applies and not current.run(ctx.drive_root, "run").patch_apply_pending
    assert (target / "content.txt").read_bytes() == b"base\n"
