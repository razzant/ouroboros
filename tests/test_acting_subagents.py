"""Tests for mutative ("acting") subagents: authority, fail-closed profile,
registry gating, worktree lifecycle, spawn gate, and patch integration."""

import json
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.contracts.task_constraint import (
    VALID_WRITE_SURFACES,
    TaskConstraint,
    normalize_task_constraint,
)
from ouroboros.tool_access import active_tool_profile
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, ACTING_SUBAGENT_TOOL_NAMES
from ouroboros.runtime_mode_policy import mode_allows_protected_write
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros import subagent_worktrees as sw


def _git(repo, *args, check=True):
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=check)


def _init_repo(path: pathlib.Path, files: dict) -> str:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")
    for rel, content in files.items():
        fp = path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")
    return _git(path, "rev-parse", "HEAD").stdout.strip()


# --------------------------------------------------------------------------- #
# 1. TaskConstraint normalization
# --------------------------------------------------------------------------- #
def test_acting_constraint_normalize_forces_invariants():
    c = normalize_task_constraint({
        "mode": "acting_subagent",
        "surface": "self_worktree",
        "write_root": "/tmp/wt",
        "base_sha": "abc",
        "protected_paths_grant": True,
        "external_tool_grants": ["mcp_x", "", "  "],
        "allow_enable": True,
        "allow_review": True,
    })
    assert c.mode == ACTING_SUBAGENT_MODE
    assert c.surface == "self_worktree"
    assert c.allow_enable is False and c.allow_review is False
    assert c.parent_only_commit is True
    assert c.external_tool_grants == ("mcp_x",)
    assert c.return_kind == "workspace_patch"


def test_acting_constraint_invalid_surface_blanked():
    c = normalize_task_constraint({"mode": "acting_subagent", "surface": "bogus"})
    assert c.mode == ACTING_SUBAGENT_MODE
    assert c.surface == ""


def test_acting_constraint_instance_repins_flags():
    c = normalize_task_constraint(TaskConstraint(mode="acting_subagent", allow_enable=True, allow_review=True))
    assert c.allow_enable is False and c.allow_review is False and c.parent_only_commit is True


# --------------------------------------------------------------------------- #
# 2. Fail-closed profile (the core safety invariant)
# --------------------------------------------------------------------------- #
def _profile_ctx(tmp_path, *, constraint=None, metadata=None):
    repo = tmp_path / "repo"; repo.mkdir(exist_ok=True)
    drive = tmp_path / "data"; drive.mkdir(exist_ok=True)
    return ToolContext(
        repo_dir=repo, drive_root=drive,
        task_constraint=constraint,
        task_metadata=metadata or {},
    )


def test_profile_acting_valid_surface(tmp_path):
    ctx = _profile_ctx(tmp_path, constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree"))
    assert active_tool_profile(ctx) == "acting_subagent"


def test_profile_acting_invalid_surface_fails_closed(tmp_path):
    ctx = _profile_ctx(tmp_path, constraint=TaskConstraint(mode="acting_subagent", surface=""))
    assert active_tool_profile(ctx) == "local_readonly_subagent"


def test_profile_readonly(tmp_path):
    ctx = _profile_ctx(tmp_path, constraint=TaskConstraint(mode="local_readonly_subagent"))
    assert active_tool_profile(ctx) == "local_readonly_subagent"


def test_profile_subagent_without_constraint_fails_closed(tmp_path):
    # Delegated subagent with a missing constraint must NEVER inherit self_modification.
    ctx = _profile_ctx(tmp_path, constraint=None, metadata={"delegation_role": "subagent"})
    assert active_tool_profile(ctx) == "local_readonly_subagent"


def test_profile_normal_task_is_self_modification(tmp_path):
    ctx = _profile_ctx(tmp_path, constraint=None, metadata={})
    assert active_tool_profile(ctx) == "self_modification"


# --------------------------------------------------------------------------- #
# 3. Registry gating for acting subagents
# --------------------------------------------------------------------------- #
def _acting_registry(tmp_path, *, surface="self_worktree", grant=False, grants=()):
    repo = tmp_path / "repo"; repo.mkdir(exist_ok=True)
    drive = tmp_path / "data"; drive.mkdir(exist_ok=True)
    worktree = tmp_path / "wt"; worktree.mkdir(exist_ok=True)
    ctx = ToolContext(
        repo_dir=repo, drive_root=drive,
        workspace_root=str(worktree), workspace_mode=surface,
        task_constraint=TaskConstraint(
            mode="acting_subagent", surface=surface, write_root=str(worktree),
            protected_paths_grant=grant, external_tool_grants=tuple(grants),
        ),
    )
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)
    reg._ctx = ctx
    return reg, ctx, worktree


def test_acting_blocks_commit(tmp_path):
    reg, _ctx, _wt = _acting_registry(tmp_path)
    out = reg.execute("commit_reviewed", {"message": "x"})
    assert "ACTING_SUBAGENT_BLOCKED" in out


def test_acting_allows_write_in_worktree(tmp_path):
    reg, _ctx, worktree = _acting_registry(tmp_path)
    out = reg.execute("write_file", {"root": "active_workspace", "path": "feature.txt", "content": "hi\n"})
    assert "ACTING_SUBAGENT_BLOCKED" not in out
    assert (worktree / "feature.txt").read_text(encoding="utf-8") == "hi\n"


def test_acting_protected_write_blocked_without_pro_grant(tmp_path):
    reg, _ctx, _wt = _acting_registry(tmp_path, surface="self_worktree", grant=False)
    out = reg.execute("write_file", {"root": "active_workspace", "path": "ouroboros/safety.py", "content": "x\n"})
    # advanced (default) + no grant => protected block, regardless of grant.
    assert "protected" in out.lower() or "PROTECTED" in out


def test_protected_write_guard_covers_redundant_root_prefix(tmp_path, monkeypatch):
    """v6.35.0 security: repo_path normalizes a redundant root-basename prefix
    ('repo/BIBLE.md' -> 'BIBLE.md'), so the protected-write guard MUST check the
    same normalized form — else a redundant-prefix path bypasses the constitution
    guard. Regression for the T2 path-normalization interaction."""
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    base = tmp_path / "repo"
    (base / "ouroboros").mkdir(parents=True)
    (base / "BIBLE.md").write_text("# real\n", encoding="utf-8")
    (base / "ouroboros" / "safety.py").write_text("# real\n", encoding="utf-8")
    reg = ToolRegistry(repo_dir=base, drive_root=tmp_path / "data")
    (tmp_path / "data").mkdir(exist_ok=True)
    reg.set_context(ToolContext(repo_dir=base, drive_root=tmp_path / "data"))

    for path in ("repo/BIBLE.md", "BIBLE.md", "repo/ouroboros/safety.py", "ouroboros/safety.py"):
        out = reg.execute("write_file", {"root": "active_workspace", "path": path, "content": "HACK"})
        assert "CORE_PROTECTION_BLOCKED" in out or "protected" in out.lower(), f"{path} not blocked: {out[:80]}"
    # The real protected files were never overwritten.
    assert (base / "BIBLE.md").read_text() == "# real\n"
    assert (base / "ouroboros" / "safety.py").read_text() == "# real\n"
    # A genuinely non-protected redundant-prefix write still succeeds.
    ok = reg.execute("write_file", {"root": "active_workspace", "path": "repo/notes.txt", "content": "hi\n"})
    assert "Written" in ok or "written" in ok


def test_shrink_guard_covers_redundant_root_prefix(tmp_path, monkeypatch):
    """v6.35.0 root-fix: the dispatch normalizes args['path'], so the accidental-
    truncation shrink guard (which checks `git ls-files` tracked status) stays
    active for a redundant-root-prefix write to an already-tracked file — it must
    not be silently disabled by the path desync."""
    import subprocess

    from ouroboros.tools.registry import ToolContext, ToolRegistry

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    base = tmp_path / "repo"
    base.mkdir()
    subprocess.run(["git", "init"], cwd=base, check=True, capture_output=True)
    (base / "large.py").write_text("\n".join(f"line {i}" for i in range(300)), encoding="utf-8")
    subprocess.run(["git", "add", "large.py"], cwd=base, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "i"],
                   cwd=base, check=True, capture_output=True)
    reg = ToolRegistry(repo_dir=base, drive_root=tmp_path / "d")
    (tmp_path / "d").mkdir(exist_ok=True)
    reg.set_context(ToolContext(repo_dir=base, drive_root=tmp_path / "d"))

    out = reg.execute("write_file", {"root": "active_workspace", "path": "repo/large.py", "content": "tiny\n"})
    assert "WRITE_BLOCKED" in out or "%" in out  # shrink guard fired
    assert len((base / "large.py").read_text(encoding="utf-8")) > 1000  # original NOT truncated


def test_acting_tool_visibility_is_acting_set(tmp_path):
    reg, _ctx, _wt = _acting_registry(tmp_path)
    names = set(reg.initial_tool_names())
    assert names == set(ACTING_SUBAGENT_TOOL_NAMES)
    assert "commit_reviewed" not in names
    assert "integrate_subagent_patch" in names


# --------------------------------------------------------------------------- #
# 4. Worktree lifecycle
# --------------------------------------------------------------------------- #
def test_worktree_provision_remove(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    head = _init_repo(repo, {"a.txt": "hi\n"})
    data = tmp_path / "data"; data.mkdir()
    wtroot = tmp_path / "wtroot"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    h = sw.provision_worktree(repo_dir=repo, task_id="t1", worktree_root=wtroot, data_dir=data)
    assert pathlib.Path(h.path).exists()
    assert h.base_sha == head
    assert sw.list_worktrees(data_dir=data)[0]["task_id"] == "t1"
    assert sw.remove_worktree(task_id="t1", worktree_root=wtroot, data_dir=data)
    assert not pathlib.Path(h.path).exists()
    assert sw.list_worktrees(data_dir=data) == []


def test_worktree_root_isolation_guard(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    with pytest.raises(ValueError):
        sw.provision_worktree(repo_dir=repo, task_id="t", worktree_root=repo / "inside", data_dir=tmp_path / "d")


def test_worktree_prune_guards_outside_root(tmp_path):
    # A corrupt registry entry pointing OUTSIDE the worktree root must never cause deletion.
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    data = tmp_path / "data"; data.mkdir()
    wtroot = tmp_path / "wtroot"
    outside = tmp_path / "precious"; outside.mkdir()
    (outside / "keep.txt").write_text("x", encoding="utf-8")
    sw._save_registry(
        [{"task_id": "evil", "path": str(outside), "branch": "", "repo_dir": str(repo), "created_at": 0.0}],
        data_dir=data,
    )
    sw.prune_orphans(worktree_root=wtroot, data_dir=data, retention_days=0)
    assert outside.exists() and (outside / "keep.txt").exists()  # guard prevented deletion


def test_worktree_prune_missing(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    data = tmp_path / "data"; data.mkdir()
    wtroot = tmp_path / "wtroot"
    h = sw.provision_worktree(repo_dir=repo, task_id="t2", worktree_root=wtroot, data_dir=data)
    import shutil
    shutil.rmtree(h.path)
    res = sw.prune_orphans(worktree_root=wtroot, data_dir=data, retention_days=9999)
    assert res["removed"] == 1 and res["kept"] == 0


# --------------------------------------------------------------------------- #
# 5. control._build_acting_constraint
# --------------------------------------------------------------------------- #
def test_build_acting_constraint_toggle(monkeypatch):
    from ouroboros.tools.control import _build_acting_constraint
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "false")
    err = _build_acting_constraint(write_surface="self_worktree", write_root="", protected_paths_grant=False, external_tool_grants=None, parent_workspace_root="")
    assert isinstance(err, str) and "MUTATIVE_SUBAGENTS_DISABLED" in err
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    c = _build_acting_constraint(write_surface="self_worktree", write_root="", protected_paths_grant=False, external_tool_grants=["x"], parent_workspace_root="")
    assert isinstance(c, dict) and c["mode"] == ACTING_SUBAGENT_MODE and c["external_tool_grants"] == ["x"]


def test_build_acting_constraint_bad_surface(monkeypatch):
    from ouroboros.tools.control import _build_acting_constraint
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    err = _build_acting_constraint(write_surface="bogus", write_root="", protected_paths_grant=False, external_tool_grants=None, parent_workspace_root="")
    assert isinstance(err, str) and "write_surface" in err


def test_build_acting_external_requires_root(monkeypatch):
    from ouroboros.tools.control import _build_acting_constraint
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    err = _build_acting_constraint(write_surface="external_workspace", write_root="", protected_paths_grant=False, external_tool_grants=None, parent_workspace_root="")
    assert isinstance(err, str) and "external_workspace" in err


# --------------------------------------------------------------------------- #
# 6. events._resolve_subagent_constraint (authoritative gate + provisioning)
# --------------------------------------------------------------------------- #
def test_resolve_readonly_passthrough(tmp_path):
    from supervisor.events import _resolve_subagent_constraint
    ctx = SimpleNamespace(REPO_DIR=tmp_path / "repo")
    c, wr, wm, detail = _resolve_subagent_constraint(
        ctx, tid="t", requested_constraint={"mode": "local_readonly_subagent"},
        workspace_root="", workspace_mode="", base_sha="", parent_task_id="",
    )
    assert c["mode"] == "local_readonly_subagent" and detail == ""


def test_resolve_acting_disabled_rejects(tmp_path, monkeypatch):
    from supervisor.events import _resolve_subagent_constraint
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "false")
    ctx = SimpleNamespace(REPO_DIR=tmp_path / "repo")
    c, wr, wm, detail = _resolve_subagent_constraint(
        ctx, tid="t", requested_constraint={"mode": "acting_subagent", "surface": "self_worktree"},
        workspace_root="", workspace_mode="", base_sha="", parent_task_id="",
    )
    assert c["mode"] == "local_readonly_subagent" and "disabled" in detail.lower()


def test_resolve_acting_self_worktree_provisions(tmp_path, monkeypatch):
    from supervisor.events import _resolve_subagent_constraint
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "wtroot"))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    ctx = SimpleNamespace(REPO_DIR=repo)
    c, wr, wm, detail = _resolve_subagent_constraint(
        ctx, tid="tw", requested_constraint={"mode": "acting_subagent", "surface": "self_worktree"},
        workspace_root="", workspace_mode="", base_sha="", parent_task_id="p",
    )
    assert detail == ""
    assert c["mode"] == "acting_subagent" and c["surface"] == "self_worktree"
    assert wr and pathlib.Path(wr).exists() and wm == "self_worktree"
    assert c["write_root"] == wr and c["base_sha"]


def test_reject_cleans_up_provisioned_worktree(tmp_path, monkeypatch):
    from supervisor.events import _resolve_subagent_constraint, _cleanup_rejected_worktree
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "wtroot"))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    ctx = SimpleNamespace(REPO_DIR=repo)
    c, wr, wm, detail = _resolve_subagent_constraint(
        ctx, tid="leak1", requested_constraint={"mode": "acting_subagent", "surface": "self_worktree"},
        workspace_root="", workspace_mode="", base_sha="", parent_task_id="p",
    )
    assert pathlib.Path(wr).exists()  # provisioned
    # A later rejection gate fires -> _reject_schedule_task tears the worktree down.
    _cleanup_rejected_worktree("leak1", {"task_constraint": c})
    assert not pathlib.Path(wr).exists()  # no leak


# --------------------------------------------------------------------------- #
# 7. integrate_subagent_patch
# --------------------------------------------------------------------------- #
def _make_child_patch(target_repo: pathlib.Path, drive: pathlib.Path, child_id: str, rel: str, new_content: str, parent_task_id: str = "parent1", surface: str = ""):
    """Produce a real workspace.patch + manifest + lineage task_result for ``rel``."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_results import task_result_path
    from hashlib import sha256
    (target_repo / rel).parent.mkdir(parents=True, exist_ok=True)
    original = (target_repo / rel).read_text(encoding="utf-8") if (target_repo / rel).exists() else ""
    (target_repo / rel).write_text(new_content, encoding="utf-8")
    patch = _git(target_repo, "diff", "--binary", "HEAD", "--").stdout
    # revert working tree so the patch can be applied fresh by the tool
    _git(target_repo, "checkout", "--", rel) if original else (target_repo / rel).unlink()
    art = task_artifact_dir_path(drive, child_id, create=True)
    # Mirror production (headless.write workspace.patch): write the exact bytes we
    # hash. write_text() would translate "\n" -> "\r\n" on Windows, so the file's
    # sha256 (read back as bytes by the integrate tool) would diverge from the
    # manifest digest and trip INTEGRATE_PATCH_CORRUPT. Binary write keeps parity.
    patch_bytes = patch.encode("utf-8")
    (art / "workspace.patch").write_bytes(patch_bytes)
    digest = sha256(patch_bytes).hexdigest()
    manifest = {
        "schema_version": 1, "status": "ready_with_changes",
        "patch_name": "workspace.patch", "sha256": digest,
        "tracked_changed": [rel] if original else [], "untracked_included": [] if original else [rel],
        "diffstat": f"{rel} | 1 +",
    }
    (art / "workspace_patch.json").write_text(json.dumps(manifest), encoding="utf-8")
    tr = task_result_path(drive, child_id)
    tr.parent.mkdir(parents=True, exist_ok=True)
    result = {"id": child_id, "parent_task_id": parent_task_id, "status": "done"}
    if surface:
        result["task_constraint"] = {"mode": "acting_subagent", "surface": surface}
    tr.write_text(json.dumps(result), encoding="utf-8")
    return art


def _make_child_delete_patch(target_repo: pathlib.Path, drive: pathlib.Path, child_id: str, rel: str, parent_task_id: str = "parent1", surface: str = ""):
    """Produce a real workspace.patch that deletes ``rel``."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_results import task_result_path
    from hashlib import sha256

    (target_repo / rel).unlink()
    patch = _git(target_repo, "diff", "--binary", "HEAD", "--").stdout
    _git(target_repo, "checkout", "--", rel)
    art = task_artifact_dir_path(drive, child_id, create=True)
    patch_bytes = patch.encode("utf-8")
    (art / "workspace.patch").write_bytes(patch_bytes)
    digest = sha256(patch_bytes).hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "ready_with_changes",
        "patch_name": "workspace.patch",
        "sha256": digest,
        "tracked_changed": [rel],
        "untracked_included": [],
        "diffstat": f"{rel} | 1 -",
    }
    (art / "workspace_patch.json").write_text(json.dumps(manifest), encoding="utf-8")
    tr = task_result_path(drive, child_id)
    tr.parent.mkdir(parents=True, exist_ok=True)
    result = {"id": child_id, "parent_task_id": parent_task_id, "status": "done"}
    if surface:
        result["task_constraint"] = {"mode": "acting_subagent", "surface": surface}
    tr.write_text(json.dumps(result), encoding="utf-8")
    return art


def _record_child_workspace_root(drive: pathlib.Path, child_id: str, workspace: pathlib.Path) -> None:
    from ouroboros.task_results import task_result_path

    child_result_path = task_result_path(drive, child_id)
    child_result = json.loads(child_result_path.read_text(encoding="utf-8"))
    child_result["workspace_root"] = str(workspace)
    child_result.setdefault("task_constraint", {})["write_root"] = str(workspace)
    child_result_path.write_text(json.dumps(child_result), encoding="utf-8")


def _integrate_ctx(target_repo, drive, **constraint_kw):
    tc = TaskConstraint(**constraint_kw) if constraint_kw else None
    return ToolContext(repo_dir=target_repo, drive_root=drive, task_constraint=tc, task_id="parent1")


def test_integrate_apply_happy(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_patch(repo, drive, "child1", "a.txt", "hi\nworld\n")
    ctx = _integrate_ctx(repo, drive)
    out = _integrate_subagent_patch(ctx, task_id="child1", reason="best of N")
    assert "Integrated subagent patch" in out
    assert (repo / "a.txt").read_text(encoding="utf-8") == "hi\nworld\n"
    # verdict artifact written
    from ouroboros.artifacts import task_artifact_dir_path
    vp = task_artifact_dir_path(drive, "parent1") / "subagent_patch_verdict_child1.json"
    assert vp.exists() and json.loads(vp.read_text())["outcome"] == "applied"


def test_integrate_reject_records_verdict(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_patch(repo, drive, "child2", "a.txt", "hi\nx\n")
    ctx = _integrate_ctx(repo, drive)
    out = _integrate_subagent_patch(ctx, task_id="child2", decision="reject", reason="worse")
    assert "Rejected subagent patch" in out
    assert (repo / "a.txt").read_text(encoding="utf-8") == "hi\n"  # unchanged


def test_integrate_protected_blocked_in_advanced(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch
    repo = tmp_path / "repo"
    _init_repo(repo, {"ouroboros/safety.py": "X = 1\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_patch(repo, drive, "child3", "ouroboros/safety.py", "X = 2\n")
    ctx = _integrate_ctx(repo, drive)
    out = _integrate_subagent_patch(ctx, task_id="child3")
    assert "protected" in out.lower() or "PROTECTED" in out
    assert (repo / "ouroboros/safety.py").read_text(encoding="utf-8") == "X = 1\n"  # unchanged


def test_integrate_corrupt_sha_refused(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    drive = tmp_path / "data"; drive.mkdir()
    art = _make_child_patch(repo, drive, "child4", "a.txt", "hi\nz\n")
    manifest = json.loads((art / "workspace_patch.json").read_text())
    manifest["sha256"] = "deadbeef"
    (art / "workspace_patch.json").write_text(json.dumps(manifest), encoding="utf-8")
    ctx = _integrate_ctx(repo, drive)
    out = _integrate_subagent_patch(ctx, task_id="child4")
    assert "INTEGRATE_PATCH_CORRUPT" in out


def test_mode_allows_protected_write_matrix():
    assert mode_allows_protected_write("pro") is True
    assert mode_allows_protected_write("advanced") is False
    assert mode_allows_protected_write("light") is False


# --------------------------------------------------------------------------- #
# 8. Adversarial round-1 fixes: ext/MCP schema deny-by-default + top-only target
# --------------------------------------------------------------------------- #
def test_acting_schemas_subset_of_acting_set(tmp_path):
    reg, _ctx, _wt = _acting_registry(tmp_path, grants=("mcp_foo",))
    names = {s["function"]["name"] for s in reg.schemas()}
    # No first-party tool outside the acting set; ungranted ext/MCP never leak.
    assert names <= set(ACTING_SUBAGENT_TOOL_NAMES)
    assert reg._acting_tool_grants() == {"mcp_foo"}


def test_integrate_acting_rejects_foreign_target_root(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch
    live = tmp_path / "live"
    _init_repo(live, {"a.txt": "hi\n"})
    worktree = tmp_path / "wt"
    _init_repo(worktree, {"a.txt": "hi\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_patch(worktree, drive, "gc1", "a.txt", "hi\nx\n", parent_task_id="acting_parent")
    ctx = ToolContext(
        repo_dir=live, drive_root=drive, workspace_root=str(worktree), workspace_mode="self_worktree",
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree", write_root=str(worktree)),
        task_id="acting_parent",
    )
    out = _integrate_subagent_patch(ctx, task_id="gc1", target_root=str(live))
    assert "INTEGRATE_TARGET_FORBIDDEN" in out
    assert (live / "a.txt").read_text(encoding="utf-8") == "hi\n"  # live repo untouched


def test_integrate_self_worktree_patch_refused_under_external_workspace(tmp_path):
    """v6.56.0 fail-closed category guard: a self_worktree child's patch targets the
    Ouroboros SYSTEM repo; an external-workspace parent must not 3-way-apply it into
    the task workspace (wrong repository)."""
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch
    system_repo = tmp_path / "system_repo"
    _init_repo(system_repo, {"a.txt": "hi\n"})
    workspace = tmp_path / "workspace"
    _init_repo(workspace, {"app.txt": "x\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_patch(system_repo, drive, "childw", "a.txt", "hi\npatched\n", surface="self_worktree")
    ctx = ToolContext(
        repo_dir=system_repo, drive_root=drive, task_id="parent1",
        workspace_root=str(workspace), workspace_mode="external",
    )
    out = _integrate_subagent_patch(ctx, task_id="childw")
    assert "INTEGRATE_SELF_WORKTREE_UNDER_WORKSPACE" in out
    assert not (workspace / "a.txt").exists()  # nothing applied into the workspace
    # A nested acting parent integrating into its OWN self_worktree stays allowed
    # (top-only routing): same child surface, workspace_mode=self_worktree.
    worktree = tmp_path / "wt"
    _init_repo(worktree, {"a.txt": "hi\n"})
    _make_child_patch(worktree, drive, "childn", "a.txt", "hi\nnested\n", parent_task_id="acting_parent", surface="self_worktree")
    nested_ctx = ToolContext(
        repo_dir=system_repo, drive_root=drive, task_id="acting_parent",
        workspace_root=str(worktree), workspace_mode="self_worktree",
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree", write_root=str(worktree)),
    )
    out2 = _integrate_subagent_patch(nested_ctx, task_id="childn")
    assert "INTEGRATE_SELF_WORKTREE_UNDER_WORKSPACE" not in out2


def test_acting_no_workspace_blocks_live_repo_write_and_shell(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    drive = tmp_path / "data"; drive.mkdir()
    # Fail-closed: an acting child whose isolated workspace did NOT resolve (no
    # workspace_root) — active_workspace/system_repo would fall back to the live repo.
    ctx = ToolContext(
        repo_dir=repo, drive_root=drive,
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree"),
    )
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)
    reg._ctx = ctx
    out = reg.execute("write_file", {"root": "active_workspace", "path": "x.txt", "content": "y\n"})
    assert "ACTING_NO_WORKSPACE_BLOCKED" in out
    assert not (repo / "x.txt").exists()
    assert "ACTING_NO_WORKSPACE_BLOCKED" in reg.execute("run_command", {"cmd": "echo hi > z.txt"})
    # claude_code_edit is not in the acting tool set -> blocked by the acting hard-block.
    assert "ACTING_SUBAGENT_BLOCKED" in reg.execute("claude_code_edit", {"cwd": ".", "instructions": "x"})
    assert "ACTING_NO_WORKSPACE_BLOCKED" in reg.execute("start_service", {"name": "svc", "cmd": "sleep 1"})


def test_integrate_acting_into_own_worktree_ok(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch
    live = tmp_path / "live"
    _init_repo(live, {"a.txt": "hi\n"})
    worktree = tmp_path / "wt"
    _init_repo(worktree, {"a.txt": "hi\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_patch(worktree, drive, "gc2", "a.txt", "hi\ny\n", parent_task_id="acting_parent2")
    ctx = ToolContext(
        repo_dir=live, drive_root=drive, workspace_root=str(worktree), workspace_mode="self_worktree",
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree", write_root=str(worktree)),
        task_id="acting_parent2",
    )
    out = _integrate_subagent_patch(ctx, task_id="gc2")  # no target_root -> own worktree
    assert "Integrated subagent patch" in out
    assert (worktree / "a.txt").read_text(encoding="utf-8") == "hi\ny\n"
    assert (live / "a.txt").read_text(encoding="utf-8") == "hi\n"  # live untouched (top-only)


def test_integrate_external_workspace_verifies_shared_files_without_reapplying(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    workspace = tmp_path / "workspace"
    _init_repo(workspace, {"a.txt": "hi\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_patch(workspace, drive, "ext1", "a.txt", "hi\nexternal\n", parent_task_id="parent-ext", surface="external_workspace")
    # Production external_workspace children write into the same directory the parent
    # can inspect; integration should verify, not apply the patch a second time.
    (workspace / "a.txt").write_text("hi\nexternal\n", encoding="utf-8")
    _record_child_workspace_root(drive, "ext1", workspace)
    ctx = ToolContext(
        repo_dir=tmp_path / "live",
        drive_root=drive,
        workspace_root=str(workspace),
        workspace_mode="external",
        task_id="parent-ext",
    )

    out = _integrate_subagent_patch(ctx, task_id="ext1")

    assert "Verified external_workspace child" in out
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "hi\nexternal\n"
    verdict = json.loads((drive / "task_results" / "artifacts" / "parent-ext" / "subagent_patch_verdict_ext1.json").read_text(encoding="utf-8"))
    assert verdict["outcome"] == "verified_shared_workspace"
    assert verdict["applied"] is False


def test_integrate_external_workspace_rejects_existing_but_mismatched_files(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    workspace = tmp_path / "workspace"
    _init_repo(workspace, {"a.txt": "hi\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_patch(workspace, drive, "ext-drift", "a.txt", "hi\nexternal\n", parent_task_id="parent-ext", surface="external_workspace")
    (workspace / "a.txt").write_text("hi\ndrifted\n", encoding="utf-8")
    _record_child_workspace_root(drive, "ext-drift", workspace)
    ctx = ToolContext(
        repo_dir=tmp_path / "live",
        drive_root=drive,
        workspace_root=str(workspace),
        workspace_mode="external",
        task_id="parent-ext",
    )

    out = _integrate_subagent_patch(ctx, task_id="ext-drift")

    assert "INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH" in out
    verdict = json.loads((drive / "task_results" / "artifacts" / "parent-ext" / "subagent_patch_verdict_ext-drift.json").read_text(encoding="utf-8"))
    assert verdict["outcome"] == "shared_workspace_mismatch"
    assert verdict["applied"] is False


def test_integrate_external_workspace_requires_parent_active_workspace(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    system_repo = tmp_path / "system"
    _init_repo(system_repo, {"README.md": "system\n"})
    workspace = tmp_path / "project"
    _init_repo(workspace, {"a.txt": "hi\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_patch(workspace, drive, "ext-project", "a.txt", "hi\nexternal\n", parent_task_id="parent-ext", surface="external_workspace")
    (workspace / "a.txt").write_text("hi\nexternal\n", encoding="utf-8")
    _record_child_workspace_root(drive, "ext-project", workspace)
    ctx = ToolContext(repo_dir=system_repo, drive_root=drive, task_id="parent-ext")

    out = _integrate_subagent_patch(ctx, task_id="ext-project")

    assert "INTEGRATE_EXTERNAL_WORKSPACE_PARENT_MISSING" in out
    verdict = json.loads((drive / "task_results" / "artifacts" / "parent-ext" / "subagent_patch_verdict_ext-project.json").read_text(encoding="utf-8"))
    assert verdict["outcome"] == "shared_workspace_parent_missing"
    assert verdict["target_root"] == str(system_repo.resolve(strict=False))
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "hi\nexternal\n"


def test_integrate_external_workspace_rejects_child_root_outside_parent_workspace(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    system_repo = tmp_path / "system"
    _init_repo(system_repo, {"README.md": "system\n"})
    parent_workspace = tmp_path / "parent-project"
    _init_repo(parent_workspace, {"a.txt": "parent\n"})
    child_workspace = tmp_path / "other-project"
    _init_repo(child_workspace, {"a.txt": "hi\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_patch(child_workspace, drive, "ext-other", "a.txt", "hi\nexternal\n", parent_task_id="parent-ext", surface="external_workspace")
    (child_workspace / "a.txt").write_text("hi\nexternal\n", encoding="utf-8")
    _record_child_workspace_root(drive, "ext-other", child_workspace)
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=drive,
        workspace_root=str(parent_workspace),
        workspace_mode="external",
        task_id="parent-ext",
    )

    out = _integrate_subagent_patch(ctx, task_id="ext-other")

    assert "INTEGRATE_EXTERNAL_WORKSPACE_TARGET_MISMATCH" in out
    assert (parent_workspace / "a.txt").read_text(encoding="utf-8") == "parent\n"
    verdict = json.loads((drive / "task_results" / "artifacts" / "parent-ext" / "subagent_patch_verdict_ext-other.json").read_text(encoding="utf-8"))
    assert verdict["outcome"] == "shared_workspace_target_mismatch"
    assert verdict["target_root"] == str(parent_workspace.resolve(strict=False))


def test_integrate_external_workspace_verifies_deletions(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    workspace = tmp_path / "workspace"
    _init_repo(workspace, {"obsolete.txt": "remove me\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_delete_patch(workspace, drive, "ext-delete", "obsolete.txt", parent_task_id="parent-ext", surface="external_workspace")
    (workspace / "obsolete.txt").unlink()
    _record_child_workspace_root(drive, "ext-delete", workspace)
    ctx = ToolContext(
        repo_dir=tmp_path / "live",
        drive_root=drive,
        workspace_root=str(workspace),
        workspace_mode="external",
        task_id="parent-ext",
    )

    out = _integrate_subagent_patch(ctx, task_id="ext-delete")

    assert "Verified external_workspace child" in out
    verdict = json.loads((drive / "task_results" / "artifacts" / "parent-ext" / "subagent_patch_verdict_ext-delete.json").read_text(encoding="utf-8"))
    assert verdict["outcome"] == "verified_shared_workspace"
    assert verdict["files"] == ["obsolete.txt"]


def test_integrate_external_workspace_files_derived_from_patch_not_manifest(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    workspace = tmp_path / "workspace"
    _init_repo(workspace, {"a.txt": "hi\n"})
    drive = tmp_path / "data"; drive.mkdir()
    art = _make_child_patch(workspace, drive, "ext-lie", "a.txt", "hi\nexternal\n", parent_task_id="parent-ext", surface="external_workspace")
    manifest = json.loads((art / "workspace_patch.json").read_text(encoding="utf-8"))
    manifest["tracked_changed"] = []
    manifest["untracked_included"] = []
    (art / "workspace_patch.json").write_text(json.dumps(manifest), encoding="utf-8")
    (workspace / "a.txt").write_text("hi\nexternal\n", encoding="utf-8")
    _record_child_workspace_root(drive, "ext-lie", workspace)
    ctx = ToolContext(
        repo_dir=tmp_path / "live",
        drive_root=drive,
        workspace_root=str(workspace),
        workspace_mode="external",
        task_id="parent-ext",
    )

    out = _integrate_subagent_patch(ctx, task_id="ext-lie")

    assert "Verified external_workspace child" in out
    verdict = json.loads((drive / "task_results" / "artifacts" / "parent-ext" / "subagent_patch_verdict_ext-lie.json").read_text(encoding="utf-8"))
    assert verdict["files"] == ["a.txt"]


# --------------------------------------------------------------------------- #
# 9. Triad+scope round-1 fixes: lineage, strict bool, owner toggle plumbing
# --------------------------------------------------------------------------- #
def test_acting_protected_grant_strict_bool():
    # String "false" must NOT grant protected authority (strict parse via normalize).
    c = normalize_task_constraint({"mode": "acting_subagent", "surface": "self_worktree", "protected_paths_grant": "false"})
    assert c.protected_paths_grant is False
    c2 = normalize_task_constraint({"mode": "acting_subagent", "surface": "self_worktree", "protected_paths_grant": "true"})
    assert c2.protected_paths_grant is True


def test_allow_mutative_settings_applied(monkeypatch):
    from ouroboros.config import apply_settings_to_env, get_allow_mutative_subagents
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    assert get_allow_mutative_subagents() is True
    # A persisted owner setting must take effect (key is in apply_settings_to_env env_keys).
    apply_settings_to_env({"OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS": "false"})
    assert get_allow_mutative_subagents() is False


def test_integrate_lineage_forbidden_for_non_child(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    drive = tmp_path / "data"; drive.mkdir()
    # Child task result claims a DIFFERENT parent than this ctx.task_id ("parent1").
    _make_child_patch(repo, drive, "gcX", "a.txt", "hi\nq\n", parent_task_id="SOMEONE_ELSE")
    ctx = _integrate_ctx(repo, drive)  # task_id="parent1"
    out = _integrate_subagent_patch(ctx, task_id="gcX")
    assert "INTEGRATE_LINEAGE_FORBIDDEN" in out
    assert (repo / "a.txt").read_text(encoding="utf-8") == "hi\n"  # not applied


# --------------------------------------------------------------------------- #
# 10. Triad+scope round-2/3/4 deep fixes
# --------------------------------------------------------------------------- #
def test_integrate_protected_derived_from_patch_not_manifest(tmp_path):
    # A malicious child cannot hide a protected edit by omitting it from the manifest.
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch
    repo = tmp_path / "repo"
    _init_repo(repo, {"ouroboros/safety.py": "X = 1\n"})
    drive = tmp_path / "data"; drive.mkdir()
    art = _make_child_patch(repo, drive, "evilc", "ouroboros/safety.py", "X = 2\n")
    manifest = json.loads((art / "workspace_patch.json").read_text())
    manifest["tracked_changed"] = ["README.md"]  # lie: hide the protected file
    (art / "workspace_patch.json").write_text(json.dumps(manifest), encoding="utf-8")
    ctx = _integrate_ctx(repo, drive)
    out = _integrate_subagent_patch(ctx, task_id="evilc")
    assert "protected" in out.lower() or "PROTECTED" in out  # patch-derived gate still catches it
    assert (repo / "ouroboros/safety.py").read_text(encoding="utf-8") == "X = 1\n"


def test_integrate_rejects_when_caller_has_no_task_id(tmp_path):
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_patch(repo, drive, "c5", "a.txt", "hi\nq\n")
    ctx = ToolContext(repo_dir=repo, drive_root=drive)  # no task_id -> cannot verify lineage
    out = _integrate_subagent_patch(ctx, task_id="c5")
    assert "INTEGRATE_LINEAGE_FORBIDDEN" in out


def test_profile_delegated_subagent_with_workspace_meta_fails_closed(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    drive = tmp_path / "data"; drive.mkdir()
    wt = tmp_path / "wt"; wt.mkdir()
    # Delegated subagent + workspace metadata + NO valid constraint -> read-only,
    # never workspace_task (fail-closed floor runs before the workspace branch).
    ctx = ToolContext(
        repo_dir=repo, drive_root=drive, workspace_root=str(wt), workspace_mode="external",
        task_constraint=None, task_metadata={"delegation_role": "subagent"},
    )
    assert active_tool_profile(ctx) == "local_readonly_subagent"


def test_remove_worktree_path_outside_root_guarded(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    data = tmp_path / "data"; data.mkdir()
    wtroot = tmp_path / "wtroot"
    outside = tmp_path / "precious2"; outside.mkdir()
    (outside / "k.txt").write_text("x", encoding="utf-8")
    # remove_worktree(path=<outside the configured root>) must refuse to delete.
    ok = sw.remove_worktree(path=str(outside), worktree_root=wtroot, data_dir=data)
    assert ok is False
    assert outside.exists() and (outside / "k.txt").exists()


# --------------------------------------------------------------------------- #
# 11. Triad+scope round-5 fixes: external_workspace validation + owner-only toggle
# --------------------------------------------------------------------------- #
def test_external_workspace_requires_git_outside_repo(tmp_path, monkeypatch):
    from supervisor.events import _resolve_subagent_constraint
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    ctx = SimpleNamespace(REPO_DIR=repo)
    # A non-git external workspace cannot return a workspace.patch -> rejected.
    nogit = tmp_path / "proj"; nogit.mkdir()
    c, wr, wm, detail = _resolve_subagent_constraint(
        ctx, tid="e1",
        requested_constraint={"mode": "acting_subagent", "surface": "external_workspace", "write_root": str(nogit)},
        workspace_root="", workspace_mode="", base_sha="", parent_task_id="p",
    )
    assert c["mode"] == "local_readonly_subagent" and "git working tree" in detail
    # A git working tree outside repo/data is accepted.
    proj = tmp_path / "gitproj"
    _init_repo(proj, {"x.txt": "y\n"})
    c2, wr2, wm2, detail2 = _resolve_subagent_constraint(
        ctx, tid="e2",
        requested_constraint={"mode": "acting_subagent", "surface": "external_workspace", "write_root": str(proj)},
        workspace_root="", workspace_mode="", base_sha="", parent_task_id="p",
    )
    assert detail2 == "" and c2["mode"] == "acting_subagent" and wm2 == "external_workspace" and wr2 == str(proj)
    assert c2["base_sha"] == _git(proj, "rev-parse", "HEAD").stdout.strip()


def test_external_workspace_rejects_stale_base_sha(tmp_path, monkeypatch):
    from supervisor.events import _resolve_subagent_constraint
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    proj = tmp_path / "gitproj"
    _init_repo(proj, {"x.txt": "y\n"})
    ctx = SimpleNamespace(REPO_DIR=repo)

    c, wr, wm, detail = _resolve_subagent_constraint(
        ctx,
        tid="e-stale",
        requested_constraint={
            "mode": "acting_subagent",
            "surface": "external_workspace",
            "write_root": str(proj),
            "base_sha": "0" * 40,
        },
        workspace_root="",
        workspace_mode="",
        base_sha="",
        parent_task_id="p",
    )

    assert c["mode"] == "local_readonly_subagent"
    assert wr == "" and wm == ""
    assert "base_sha is stale" in detail


def test_external_workspace_acting_base_sha_blocks_moved_head_artifact(tmp_path, monkeypatch):
    from supervisor.events import _resolve_subagent_constraint
    from ouroboros.headless import ARTIFACT_STATUS_FAILED, write_workspace_patch_artifacts

    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    proj = tmp_path / "gitproj"
    _init_repo(proj, {"x.txt": "y\n"})
    ctx = SimpleNamespace(REPO_DIR=repo)
    c, wr, wm, detail = _resolve_subagent_constraint(
        ctx,
        tid="e-artifact",
        requested_constraint={"mode": "acting_subagent", "surface": "external_workspace", "write_root": str(proj)},
        workspace_root="",
        workspace_mode="",
        base_sha="",
        parent_task_id="p",
    )
    assert detail == "" and c["base_sha"]

    (proj / "x.txt").write_text("committed by child\n", encoding="utf-8")
    _git(proj, "add", "x.txt")
    _git(proj, "commit", "-q", "-m", "child commit")
    artifacts, manifest = write_workspace_patch_artifacts(proj, tmp_path / "artifacts", task={"task_constraint": c})

    assert wr == str(proj) and wm == "external_workspace"
    assert manifest["status"] == ARTIFACT_STATUS_FAILED
    assert manifest["errors"][-1]["type"] == "workspace_head_changed"
    assert manifest["errors"][-1]["expected_head"] == c["base_sha"]
    assert not any(item["kind"] == "workspace_patch" for item in artifacts)


def test_external_workspace_unborn_base_sha_blocks_first_commit_artifact(tmp_path, monkeypatch):
    from supervisor.events import _resolve_subagent_constraint
    from ouroboros.headless import ARTIFACT_STATUS_FAILED, write_workspace_patch_artifacts

    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    proj = tmp_path / "empty-gitproj"
    proj.mkdir()
    _git(proj, "init", "-q")
    ctx = SimpleNamespace(REPO_DIR=repo)
    c, wr, wm, detail = _resolve_subagent_constraint(
        ctx,
        tid="e-unborn",
        requested_constraint={"mode": "acting_subagent", "surface": "external_workspace", "write_root": str(proj)},
        workspace_root="",
        workspace_mode="",
        base_sha="",
        parent_task_id="p",
    )
    assert detail == ""
    assert wr == str(proj) and wm == "external_workspace"
    assert c["base_sha"] == "(unborn)"

    _git(proj, "config", "user.email", "t@t")
    _git(proj, "config", "user.name", "t")
    (proj / "created.txt").write_text("first\n", encoding="utf-8")
    _git(proj, "add", "created.txt")
    _git(proj, "commit", "-q", "-m", "first commit")
    artifacts, manifest = write_workspace_patch_artifacts(proj, tmp_path / "artifacts", task={"task_constraint": c})

    assert manifest["status"] == ARTIFACT_STATUS_FAILED
    assert manifest["errors"][-1]["type"] == "workspace_head_changed"
    assert manifest["errors"][-1]["expected_head"] == "(unborn)"
    assert not any(item["kind"] == "workspace_patch" for item in artifacts)


def test_mutative_toggle_self_change_detected():
    from ouroboros.tools.registry import _detect_mutative_toggle_self_change
    assert _detect_mutative_toggle_self_change('echo true >> data/settings.json # ouroboros_allow_mutative_subagents')
    assert _detect_mutative_toggle_self_change('save_settings({"ouroboros_allow_mutative_subagents": "true"})')
    # CLI settings-set path must also be caught.
    assert _detect_mutative_toggle_self_change("ouroboros settings set ouroboros_allow_mutative_subagents true")
    assert not _detect_mutative_toggle_self_change("echo hello world")


def test_evolution_owner_control_self_change_detected():
    from ouroboros.tools.registry import _detect_evolution_owner_control_self_change as d
    assert d('echo true >> data/settings.json # ouroboros_post_task_evolution')
    assert d('save_settings({"ouroboros_post_task_evolution": "true"})')
    assert d("ouroboros settings set ouroboros_post_task_evolution true")
    # The persistent objective is owner-only too (it steers every evolution campaign).
    # Detector receives pre-lowered text (cmd_lower), mirror that here.
    assert d('curl -x post 127.0.0.1:8765/api/settings -d \'{"ouroboros_evolution_persistent_objective":"x"}\'')
    assert d('save_settings({"ouroboros_evolution_persistent_objective": "grab budget"})')
    assert not d("echo hello world")
    assert not d("ouroboros_post_task_evolution")  # key alone, no write target


def test_post_task_evolution_js_guard():
    from ouroboros.tools.browser import _blocks_post_task_evolution_js
    assert _blocks_post_task_evolution_js(
        "fetch('/api/settings', {method:'POST', body: JSON.stringify({OUROBOROS_POST_TASK_EVOLUTION: true})})"
    )
    assert _blocks_post_task_evolution_js('save_settings({"ouroboros_post_task_evolution": true})')
    assert _blocks_post_task_evolution_js(
        "fetch('/api/settings', {method:'POST', body: JSON.stringify({OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE: 'x'})})"
    )
    assert not _blocks_post_task_evolution_js("document.title")


def test_pro_acting_shell_write_outside_surface_blocked(tmp_path):
    # Even in pro mode, an acting child's write-like shell targeting outside its
    # isolated surface is blocked (no pro workspace passthrough for acting subagents).
    repo = tmp_path / "repo"; repo.mkdir()
    drive = tmp_path / "data"; drive.mkdir()
    wt = tmp_path / "wt"; wt.mkdir()
    ctx = ToolContext(
        repo_dir=repo, drive_root=drive, workspace_root=str(wt), workspace_mode="self_worktree",
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree", write_root=str(wt)),
    )
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)
    reg._ctx = ctx
    block = reg._run_shell_safety_check({"cmd": "echo x > ../outside.txt"}, "pro")
    assert block and "WORKSPACE_SHELL_BLOCKED" in block


def test_subagent_shell_secret_markers_cover_relative_paths():
    from ouroboros.tools.registry import _subagent_shell_targets_secret
    assert _subagent_shell_targets_secret("cat .env")
    assert _subagent_shell_targets_secret("cat .git/config")
    assert _subagent_shell_targets_secret("cat .git/credentials")
    assert _subagent_shell_targets_secret("cat ~/.ssh/id_rsa")
    assert not _subagent_shell_targets_secret("cat src/main.py")


def test_acting_read_schema_excludes_system_repo(tmp_path):
    reg, _ctx, _wt = _acting_registry(tmp_path)
    schemas = {s["function"]["name"]: s["function"] for s in reg.schemas()}
    rf = schemas.get("read_file")
    if rf:
        root_enum = rf["parameters"]["properties"].get("root", {}).get("enum")
        if isinstance(root_enum, list):
            assert "system_repo" not in root_enum  # matches acting _POLICY (no system_repo)


def test_acting_subagent_cannot_shell_read_secrets(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    drive = tmp_path / "data"; drive.mkdir()
    wt = tmp_path / "wt"; wt.mkdir()
    ctx = ToolContext(
        repo_dir=repo, drive_root=drive, workspace_root=str(wt), workspace_mode="self_worktree",
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree", write_root=str(wt)),
    )
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)
    reg._ctx = ctx
    block = reg._run_shell_safety_check({"cmd": "cat ~/Ouroboros/data/settings.json"}, "pro")
    assert block and "SUBAGENT_SECRET_READ_BLOCKED" in block


def test_integrate_counts_as_reviewable_effect():
    from ouroboros.outcomes import turn_has_reviewable_effects
    trace = {"tool_calls": [{"tool": "integrate_subagent_patch", "status": "ok", "args": {"task_id": "c"}}]}
    assert turn_has_reviewable_effects(trace) is True


def test_readonly_subagent_cannot_spawn_acting_child(tmp_path):
    from ouroboros.tools.control import _schedule_task
    repo = tmp_path / "repo"; repo.mkdir()
    drive = tmp_path / "data"; drive.mkdir()
    ctx = ToolContext(
        repo_dir=repo, drive_root=drive,
        task_constraint=TaskConstraint(mode="local_readonly_subagent"),
    )
    out = _schedule_task(ctx, objective="do X", expected_output="Y", write_surface="self_worktree")
    assert "MUTATIVE_SUBAGENTS_DISABLED" in out


def test_acting_schema_narrows_write_root_and_browser(tmp_path):
    reg, _ctx, _wt = _acting_registry(tmp_path)
    schemas = {s["function"]["name"]: s["function"] for s in reg.schemas()}
    wf = schemas.get("write_file")
    assert wf is not None
    root_enum = wf["parameters"]["properties"].get("root", {}).get("enum")
    if isinstance(root_enum, list):  # acting writes only its isolated surface
        assert root_enum == ["active_workspace"]
    ba = schemas.get("browser_action")
    if ba:
        action_enum = ba["parameters"]["properties"].get("action", {}).get("enum")
        if isinstance(action_enum, list):
            assert "evaluate" not in action_enum


def test_no_workspace_acting_integrate_blocked(tmp_path):
    # An acting child without a resolved workspace must not integrate into the live repo.
    repo = tmp_path / "repo"; repo.mkdir()
    drive = tmp_path / "data"; drive.mkdir()
    ctx = ToolContext(
        repo_dir=repo, drive_root=drive,
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree"),
    )
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)
    reg._ctx = ctx
    assert "ACTING_NO_WORKSPACE_BLOCKED" in reg.execute("integrate_subagent_patch", {"task_id": "x"})


def test_acting_subagent_cannot_read_secrets(tmp_path):
    # Acting children may write their surface but must NOT read owner secrets.
    from ouroboros.tools.core import _data_read
    repo = tmp_path / "repo"; repo.mkdir()
    drive = tmp_path / "data"; drive.mkdir()
    (drive / "settings.json").write_text('{"OPENAI_API_KEY": "sk-secret-xyz"}', encoding="utf-8")
    ctx = ToolContext(
        repo_dir=repo, drive_root=drive,
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree", write_root=str(tmp_path / "wt")),
    )
    out = _data_read(ctx, "settings.json")
    assert "DATA_READ_BLOCKED" in out and "sk-secret-xyz" not in out


def test_acting_subagent_keeps_workspace_access(tmp_path):
    # The strict-readonly resource block must NOT restrict acting children's worktree.
    from ouroboros.tools.core import _local_readonly_resource_block
    repo = tmp_path / "repo"; repo.mkdir()
    drive = tmp_path / "data"; drive.mkdir()
    ctx = ToolContext(
        repo_dir=repo, drive_root=drive,
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree", write_root=str(tmp_path / "wt")),
    )
    assert _local_readonly_resource_block(ctx, "active_workspace", tmp_path / "wt" / "f.txt", tmp_path / "wt", action="write") == ""


# --------------------------------------------------------------------------- #
# 14. v6.21.0: genesis surface, compare helper, unified GC retention
# --------------------------------------------------------------------------- #
def test_genesis_is_a_valid_surface():
    assert "genesis" in VALID_WRITE_SURFACES
    c = normalize_task_constraint({"mode": "acting_subagent", "surface": "genesis"})
    assert c.mode == ACTING_SUBAGENT_MODE and c.surface == "genesis"


def test_genesis_in_acting_toolset_but_not_self_worktree_discipline(tmp_path):
    # genesis is acting (recognized) but is NOT the system repo, so it must not
    # carry self_worktree protected-path discipline.
    reg, _ctx, _wt = _acting_registry(tmp_path, surface="genesis")
    assert reg._is_acting_subagent() is True
    assert reg._acting_self_worktree() is False


def test_provision_genesis_project_is_durable_and_isolated(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    proot = tmp_path / "projects"
    data = tmp_path / "data"
    h = sw.provision_genesis_project(repo_dir=repo, task_id="g1", projects_root=proot, data_dir=data)
    assert pathlib.Path(h.path).exists() and (pathlib.Path(h.path) / ".git").exists()
    assert h.base_sha  # seed commit exists
    assert sw._is_within(pathlib.Path(h.path), proot)
    # NOT added to the worktree registry (durable, not GC-tracked).
    assert all(e.get("path") != h.path for e in sw._load_registry(data))
    # remove only inside projects root; refuses arbitrary paths.
    assert sw.remove_genesis_project(str(tmp_path / "elsewhere")) is False
    assert sw.remove_genesis_project(h.path, projects_root=proot) is True
    assert not pathlib.Path(h.path).exists()


def test_force_rmtree_removes_readonly_files(tmp_path):
    # Genesis/worktree teardown must delete read-only git pack files (Windows marks
    # them read-only; shutil.rmtree(ignore_errors=True) would silently leave them).
    import os as _os, stat as _stat
    d = tmp_path / "proj" / ".git" / "objects"
    d.mkdir(parents=True)
    ro = d / "pack-readonly"
    ro.write_text("x", encoding="utf-8")
    _os.chmod(ro, _stat.S_IREAD)  # read-only, like a git pack file
    sw._force_rmtree(tmp_path / "proj")
    assert not (tmp_path / "proj").exists()


def test_provision_genesis_rejects_root_overlapping_repo(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    with pytest.raises(ValueError):
        sw.provision_genesis_project(repo_dir=repo, task_id="g2", projects_root=repo / "inside", data_dir=tmp_path / "data")


def test_resolve_acting_genesis_provisions(tmp_path, monkeypatch):
    from supervisor.events import _resolve_subagent_constraint
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    ctx = SimpleNamespace(REPO_DIR=repo)
    c, wr, wm, detail = _resolve_subagent_constraint(
        ctx, tid="gp", requested_constraint={"mode": "acting_subagent", "surface": "genesis"},
        workspace_root="", workspace_mode="", base_sha="", parent_task_id="p",
    )
    assert detail == "" and c["mode"] == "acting_subagent" and c["surface"] == "genesis"
    assert wm == "genesis" and wr and pathlib.Path(wr).exists()
    assert sw._is_within(pathlib.Path(wr), tmp_path / "projects")


def test_reject_cleans_up_provisioned_genesis(tmp_path, monkeypatch):
    from supervisor.events import _resolve_subagent_constraint, _cleanup_rejected_worktree
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    ctx = SimpleNamespace(REPO_DIR=repo)
    c, wr, wm, detail = _resolve_subagent_constraint(
        ctx, tid="gleak", requested_constraint={"mode": "acting_subagent", "surface": "genesis"},
        workspace_root="", workspace_mode="", base_sha="", parent_task_id="p",
    )
    assert detail == "" and pathlib.Path(wr).exists()
    _cleanup_rejected_worktree("gleak", {"task_constraint": c})
    assert not pathlib.Path(wr).exists()


def test_compare_subagent_patches_read_only(tmp_path):
    from ouroboros.tools.subagent_integration import _compare_subagent_patches
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n", "b.txt": "x\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_patch(repo, drive, "cA", "a.txt", "hi\nA\n")
    _make_child_patch(repo, drive, "cB", "b.txt", "x\nB\n")
    ctx = _integrate_ctx(repo, drive)
    out = _compare_subagent_patches(ctx, task_ids=["cA", "cB"])
    assert "cA" in out and "cB" in out and "```diff" in out
    # read-only: working tree unchanged
    assert (repo / "a.txt").read_text(encoding="utf-8") == "hi\n"
    assert (repo / "b.txt").read_text(encoding="utf-8") == "x\n"
    # empty arg is a clear tool error
    assert "TOOL_ARG_ERROR" in _compare_subagent_patches(ctx, task_ids=[])


def test_integrate_refuses_genesis_child(tmp_path):
    # A genesis project is a standalone deliverable, never integrated into the live body.
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch
    repo = tmp_path / "repo"
    _init_repo(repo, {"a.txt": "hi\n"})
    drive = tmp_path / "data"; drive.mkdir()
    _make_child_patch(repo, drive, "gchild", "a.txt", "hi\nnew\n", surface="genesis")
    ctx = _integrate_ctx(repo, drive)
    out = _integrate_subagent_patch(ctx, task_id="gchild")
    assert "INTEGRATE_GENESIS_FORBIDDEN" in out
    assert (repo / "a.txt").read_text(encoding="utf-8") == "hi\n"  # unchanged
    # reject is still allowed (records a verdict without applying)
    assert "Rejected" in _integrate_subagent_patch(ctx, task_id="gchild", decision="reject", reason="n/a")


def test_retention_helpers():
    from ouroboros.retention import age_cutoff, clamp_retention_days
    assert age_cutoff(7, now=1_000_000) == 1_000_000 - 7 * 86400
    assert age_cutoff(0, now=1_000_000) == 1_000_000  # explicit 0 => prune everything before now
    assert clamp_retention_days(5, default=7) == 5
    assert clamp_retention_days(0, default=7) == 7
    assert clamp_retention_days(9999, default=7) == 365
    assert clamp_retention_days("bogus", default=7) == 7


def test_get_gc_retention_days_precedence(monkeypatch):
    from ouroboros import retention
    monkeypatch.delenv("OUROBOROS_GC_RETENTION_DAYS", raising=False)
    for legacy in retention.LEGACY_RETENTION_KEYS:
        monkeypatch.delenv(legacy, raising=False)
    monkeypatch.setenv("OUROBOROS_GC_RETENTION_DAYS", "9")
    assert retention.get_gc_retention_days() == 9
    monkeypatch.delenv("OUROBOROS_GC_RETENTION_DAYS", raising=False)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_RETENTION_DAYS", "21")
    assert retention.get_gc_retention_days() == 21  # legacy fallback (no orphan)


def test_gc_retention_migration_folds_and_drops_legacy(tmp_path, monkeypatch):
    import ouroboros.config as config
    sp = tmp_path / "settings.json"
    sp.write_text(json.dumps({
        "OUROBOROS_SUBAGENT_WORKTREE_RETENTION_DAYS": 30,
        "OUROBOROS_SERVICE_LOG_RETENTION_DAYS": 14,
    }), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", sp)
    monkeypatch.setattr(config, "_SETTINGS_LOCK", pathlib.Path(str(sp) + ".lock"))
    monkeypatch.delenv("OUROBOROS_GC_RETENTION_DAYS", raising=False)
    s = config.load_settings()
    # Seeded from the worktree key (customized 30 != former default 7); no orphan.
    assert s.get("OUROBOROS_GC_RETENTION_DAYS") == 30
    assert "OUROBOROS_SUBAGENT_WORKTREE_RETENTION_DAYS" not in s
    assert "OUROBOROS_SERVICE_LOG_RETENTION_DAYS" not in s


def test_gc_migration_prefers_customized_over_default_earlier_key(tmp_path, monkeypatch):
    # Regression: a customized LATER key must win over a default EARLIER key, so a
    # customized service-log value is not dropped just because worktree is at default.
    import ouroboros.config as config
    sp = tmp_path / "settings.json"
    sp.write_text(json.dumps({
        "OUROBOROS_SUBAGENT_WORKTREE_RETENTION_DAYS": 7,    # former default (7) -> not customized
        "OUROBOROS_SERVICE_LOG_RETENTION_DAYS": 30,         # customized (former default 14) -> must win
    }), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", sp)
    monkeypatch.setattr(config, "_SETTINGS_LOCK", pathlib.Path(str(sp) + ".lock"))
    monkeypatch.delenv("OUROBOROS_GC_RETENTION_DAYS", raising=False)
    s = config.load_settings()
    assert s.get("OUROBOROS_GC_RETENTION_DAYS") == 30  # customized value preserved, not the default 7
    assert "OUROBOROS_SERVICE_LOG_RETENTION_DAYS" not in s


def test_gc_migration_all_defaults_collapse_to_unified_default(tmp_path, monkeypatch):
    # An all-defaults legacy file collapses to the unified default (service 14 -> 7 is intentional).
    import ouroboros.config as config
    sp = tmp_path / "settings.json"
    sp.write_text(json.dumps({
        "OUROBOROS_SUBAGENT_WORKTREE_RETENTION_DAYS": 7,   # former default
        "OUROBOROS_SERVICE_LOG_RETENTION_DAYS": 14,         # former default
    }), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", sp)
    monkeypatch.setattr(config, "_SETTINGS_LOCK", pathlib.Path(str(sp) + ".lock"))
    monkeypatch.delenv("OUROBOROS_GC_RETENTION_DAYS", raising=False)
    s = config.load_settings()
    # No customized value -> fall back to first present (worktree 7) == unified default.
    assert s.get("OUROBOROS_GC_RETENTION_DAYS") == 7


def test_select_subagent_constraint_read_only_token_is_readonly():
    """`write_surface='read_only'` resolves to the SAME read-only constraint as
    omitting the surface — never an acting self_worktree — giving a read-only audit
    child an explicit, provider-safe way to name its intent (P5 cancel-storm fix)."""
    from ouroboros.tools.control import _select_subagent_constraint

    baseline = _select_subagent_constraint("", "", False, [], "")  # omit = read-only
    assert isinstance(baseline, dict)
    assert baseline.get("mode")
    for surface in ("read_only", "READ_ONLY", " read_only "):
        constraint = _select_subagent_constraint(surface, "", False, [], "")
        assert isinstance(constraint, dict), surface
        assert constraint == baseline, surface
