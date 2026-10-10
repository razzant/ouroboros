"""Public consumer: admitted child B → real capture → parent verifies B, never A."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import task_artifact_dir_path, copy_directory_to_task_artifacts, copy_file_to_task_artifacts
from ouroboros.headless import finalize_task_artifacts
from ouroboros.task_results import write_task_result
from ouroboros.task_status import load_effective_task_result
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.workspace_patch_capture import write_workspace_patch_artifacts
from supervisor.events_subagent_admission import _resolve_subagent_constraint

# Actual Git subprocess fixtures follow the repository's real-process lane.
pytestmark = pytest.mark.serial


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout


def init(root):
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    (root / "a.txt").write_bytes(b"one\ntwo\nthree\n")
    git(root, "add", "a.txt")
    git(root, "commit", "-qm", "base")


@pytest.fixture
def env(tmp_path, monkeypatch):
    system, a, b, drive = [tmp_path / n for n in ("system", "a", "b", "state")]
    for root in (system, a, b):
        init(root)
    drive.mkdir()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setattr("ouroboros.tool_access._user_files_root", lambda: tmp_path)
    parent = ToolContext(repo_dir=system, drive_root=drive, task_id="parent",
                         workspace_root=a, workspace_mode="external")
    return parent, a, b, drive


def capture(parent, b, drive, *, direct=False, edit=None, outputs=("rendered",)):
    constraint, root, mode, error = _resolve_subagent_constraint(
        SimpleNamespace(REPO_DIR=parent.repo_dir, DRIVE_ROOT=drive), tid="child",
        requested_constraint={"mode": "acting_subagent", "surface": "external_workspace", "write_root": str(b)},
        workspace_root=str(b), workspace_mode="external", base_sha="", parent_task_id="parent")
    assert not error
    task = {"id": "child", "workspace_root": root, "workspace_mode": mode, "task_constraint": constraint}
    if edit:
        edit()
    else:
        (b / "a.txt").write_bytes(b"one\nchanged\nthree\n")
    write_task_result(drive, "child", "completed", result="Files written in assigned B",
                      parent_task_id="parent", root_task_id="parent", delegation_role="subagent",
                      workspace_root=root, task_constraint=constraint)
    art = task_artifact_dir_path(drive, "child", create=True)
    if direct:
        child = ToolContext(repo_dir=parent.repo_dir, drive_root=drive, task_id="child",
                            workspace_root=b, workspace_mode="external")
        for name in outputs:
            register = copy_directory_to_task_artifacts if (b / name).is_dir() else copy_file_to_task_artifacts
            register(child, b / name)
        finalize_task_artifacts(drive, task)
    else:
        _, manifest = write_workspace_patch_artifacts(b, art, task=task)
        assert manifest["status"] == "ready_with_changes", manifest
    return art


def invoke_result(parent, **args):
    registry = ToolRegistry(repo_dir=parent.repo_dir, drive_root=parent.drive_root)
    registry.set_context(parent)
    return registry.execute_result("integrate_subagent_patch", {"task_id": "child", **args})


def invoke(parent, **args):
    return invoke_result(parent, **args).text


def snapshot(root):
    files = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
             if p.is_file() and ".git" not in p.relative_to(root).parts}
    if (root / ".git").exists():
        index = Path(git(root, "rev-parse", "--git-path", "index").decode().strip())
        if not index.is_absolute():
            index = root / index
        return files, git(root, "rev-parse", "HEAD"), index.read_bytes()
    return files, None, None


def verdict(drive):
    path = task_artifact_dir_path(drive, "parent") / "subagent_patch_verdict_child.json"
    return json.loads(path.read_text(encoding="utf-8"))


def assert_unabsorbed(drive):
    assert not load_effective_task_result(drive, "child").get("child_result_disposition")


@pytest.mark.parametrize("kind", ["different_repo", "linked", "no_external", "unavailable_a", "same_root"])
def test_public_verifies_at_b_without_transfer(env, monkeypatch, kind):
    parent, a, b, drive = env
    if kind == "linked":
        subprocess.run(["git", "worktree", "add", "-q", "-b", "child-branch", str(b.parent / "linked")], cwd=a, check=True)
        b = b.parent / "linked"
    elif kind == "same_root":
        b = a
    elif kind == "no_external":
        parent.workspace_root = None
        parent.workspace_mode = ""
    elif kind == "unavailable_a":
        parent.workspace_root = a.parent / "missing"
    art = capture(parent, b, drive)
    before = snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()
    for target_args in ({}, {"target_root": str(b)}):
        out = invoke(parent, **target_args)
        assert "Verified external_workspace child" in out, out
        assert "transferred to the parent's folder" in out
        assert verdict(drive)["target_root"] == str(b.resolve())
        assert verdict(drive)["applied"] is False
        assert (snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()) == before
    assert load_effective_task_result(drive, "child")["child_result_disposition"] == "integrated"


@pytest.mark.parametrize("change", ["wrong_target", "capture_root", "result_root", "base", "missing_assignment"])
def test_identity_refusals_precede_b_verification(env, monkeypatch, change):
    parent, a, b, drive = env
    art = capture(parent, b, drive)
    manifest = json.loads((art / "workspace_patch.json").read_text())
    args = {}
    if change == "wrong_target":
        args["target_root"] = str(a)
    elif change == "capture_root":
        manifest["workspace_root"] = str(a)
    elif change == "base":
        manifest["base_head"] = "0" * 40
    else:
        row_path = drive / "task_results" / "child.json"
        row = json.loads(row_path.read_text())
        if change == "result_root":
            row["workspace_root"] = str(a)
        else:
            row.pop("workspace_root")
            row["task_constraint"].pop("write_root")
        row_path.write_text(json.dumps(row))
    (art / "workspace_patch.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("ouroboros.tools.subagent_integration._verify_shared_external_workspace",
                        lambda *a, **k: pytest.fail("B was verified before identity admission"))
    before = snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()
    # A missing assignment is not a conflicting one: nothing names a target.
    assert ("TARGET_MISSING" if change == "missing_assignment" else "TARGET_MISMATCH") in invoke(parent, **args)
    assert (snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()) == before
    assert_unabsorbed(drive)


def ceiling(parent, root, prefix):
    from ouroboros.presence_authority import (PresenceCapabilityCeiling, PresenceToolGrant,
                                             PresenceResourceGrant, presence_ceiling_payload)
    parent.task_contract = {"capability_ceiling": presence_ceiling_payload(PresenceCapabilityCeiling(
        skill_name="fixture", skill_content_hash="a" * 64, profile_fingerprint="b" * 64,
        state_fingerprint="c" * 64, selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=10,
        tool_grants=(PresenceToolGrant("integrate_subagent_patch"),),
        resource_grants=(PresenceResourceGrant(root=root, operations=("read",), path_prefix=prefix),), digest="e" * 64))}


@pytest.mark.parametrize("restriction", ["off_home", "presence", "protected_read"])
def test_read_admission_precedes_source_checks(env, monkeypatch, restriction):
    parent, a, b, drive = env
    art = capture(parent, b, drive)
    if restriction == "off_home":
        parent.workspace_root = None
        parent.workspace_mode = ""
        monkeypatch.setattr("ouroboros.tool_access._user_files_root", lambda: b.parent / "other-home")
    elif restriction == "presence":
        ceiling(parent, "active_workspace", ".")
    else:
        parent.task_contract = {"resource_policy": {"protected_artifacts": [{
            "id": "black-box", "paths": [str(b / "a.txt")], "deny": ["read_bytes"]}]}}
    monkeypatch.setattr("ouroboros.tools.subagent_integration._verify_shared_external_workspace",
                        lambda *a, **k: pytest.fail("B content verification bypassed read authority"))
    out = invoke(parent)
    assert "INTEGRATE_TARGET_FORBIDDEN" in out, out
    assert_unabsorbed(drive)
    assert (art / "workspace.patch").is_file()


def test_presence_read_grant_allows_b(env):
    parent, _, b, drive = env
    capture(parent, b, drive)
    ceiling(parent, "user_files", b.name)
    assert "Verified external_workspace child" in invoke(parent)


def test_cyber_no_external_parent_keeps_existing_off_home_read(env, monkeypatch):
    parent, _, b, drive = env
    capture(parent, b, drive)
    parent.workspace_root, parent.workspace_mode = None, ""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    monkeypatch.setattr("ouroboros.tool_access._user_files_root", lambda: b.parent / "other-home")
    assert "Verified external_workspace child" in invoke(parent)


@pytest.mark.parametrize("drift", ["patch_bytes", "content", "legacy_fields", "live_head"])
def test_capture_integrity_legacy_and_shared_head_movement(env, drift):
    parent, a, b, drive = env
    art = capture(parent, b, drive)
    if drift == "patch_bytes":
        with (art / "workspace.patch").open("ab") as stream:
            stream.write(b"corruption")
    elif drift == "content":
        (b / "a.txt").write_bytes(b"later\n")
    elif drift == "live_head":
        git(b, "add", "a.txt")
        git(b, "commit", "-qm", "shared parent commit")
    else:
        p = art / "workspace_patch.json"
        m = json.loads(p.read_text())
        m.pop("workspace_root"); m.pop("base_head")
        p.write_text(json.dumps(m))
    before = snapshot(a), snapshot(b)
    out = invoke(parent)
    assert (snapshot(a), snapshot(b)) == before
    if drift in {"patch_bytes", "content"}:
        assert "CORRUPT" in out if drift == "patch_bytes" else "WORKSPACE_MISMATCH" in out
        assert_unabsorbed(drive)
    else:
        assert "Verified external_workspace child" in out, out


@pytest.mark.parametrize("kind", ["delete", "rename", "binary"])
def test_git_path_variants_are_verified_not_applied(env, kind):
    parent, a, b, drive = env
    def edit():
        if kind == "delete":
            (b / "a.txt").unlink()
        elif kind == "rename":
            git(b, "mv", "a.txt", "renamed.txt")
            (b / "renamed.txt").write_bytes(b"one\nchanged\nthree\n")
        else:
            (b / "a.txt").write_bytes(b"binary\x00\xff")
    capture(parent, b, drive, edit=edit)
    before = snapshot(a), snapshot(b)
    assert "Verified external_workspace child" in invoke(parent)
    assert (snapshot(a), snapshot(b)) == before
    assert not verdict(drive)["applied"]


@pytest.mark.parametrize("deny_member", [False, True])
def test_directory_registered_members_at_b_and_operation_specific_denial(env, monkeypatch, deny_member):
    parent, a, b, drive = env
    # Admission chooses an ordinary non-Git B, not a second Git mechanism.
    b = b.parent / "plain"
    b.mkdir()
    def edit():
        (b / "rendered").mkdir()
        (b / "rendered" / "frame.bin").write_bytes(b"image\x00")
    art = capture(parent, b, drive, direct=True, edit=edit)
    if deny_member:
        parent.task_contract = {"resource_policy": {"protected_artifacts": [{
            "id": "black-box", "paths": [str(b / "rendered" / "frame.bin")], "deny": ["hash"]}]}}
    before = snapshot(a), snapshot(b)
    out = invoke(parent)
    assert (snapshot(a), snapshot(b)) == before
    if deny_member:
        assert "RESOURCE_POLICY_BLOCKED" in out, out
        assert_unabsorbed(drive)
    else:
        assert "Verified 1 registered file postimage(s)" in out, out
        assert verdict(drive)["outcome"] == "verified_registered_outputs"
        assert not verdict(drive)["applied"]
    assert (art / "workspace_patch.json").is_file()


def test_cooperative_target_and_wrong_explicit_target_use_common_admission(env, monkeypatch):
    parent, a, b, drive = env
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(b.parent))
    capture(parent, b, drive)
    assert "TARGET_MISMATCH" in invoke(parent, target_root=str(a))
    assert_unabsorbed(drive)
    out = invoke(parent)
    assert out.startswith("OK: cooperative no-op"), out
    assert verdict(drive)["outcome"] == "coop_already_in_tree"


def test_file_reference_postimage_hash_policy_is_checked_before_verification(env, monkeypatch):
    parent, a, b, drive = env
    monkeypatch.setattr("ouroboros.workspace_patch_capture._PATCH_FILE_REFERENCE_BYTES", 1)
    art = capture(parent, b, drive)
    manifest = json.loads((art / "workspace_patch.json").read_text())
    assert manifest["file_output_changes"]
    parent.task_contract = {"resource_policy": {"protected_artifacts": [{
        "id": "hash-only", "paths": [str(b / "a.txt")], "deny": ["hash"]}]}}
    monkeypatch.setattr("ouroboros.tools.subagent_integration._verify_shared_external_workspace",
                        lambda *a, **k: pytest.fail("postimage hashing bypassed task policy"))
    assert "RESOURCE_POLICY_BLOCKED" in invoke(parent)
    assert_unabsorbed(drive)


def test_relative_task_policy_is_not_reinterpreted_at_child_b(env):
    parent, a, b, drive = env
    capture(parent, b, drive)
    # Relative policy refers to A/a.txt, not every same-named file in B.
    parent.task_contract = {"resource_policy": {"protected_artifacts": [{
        "id": "parent-only", "paths": ["a.txt"], "deny": ["read_bytes", "hash"]}]}}
    assert "Verified external_workspace child" in invoke(parent)


def test_task_disabled_tool_cannot_gain_authority_from_child_assignment(env):
    parent, _, b, drive = env
    capture(parent, b, drive)
    parent.task_contract = {"disabled_tools": ["integrate_subagent_patch"]}
    out = invoke(parent)
    assert "Verified external_workspace child" not in out
    assert_unabsorbed(drive)


def test_symlink_path_escape_is_rejected_without_reading_destination(env, monkeypatch):
    parent, _, b, drive = env
    art = capture(parent, b, drive)
    outside = b.parent / "outside.txt"
    outside.write_bytes(b"private sentinel\n")
    (b / "a.txt").unlink()
    (b / "a.txt").symlink_to(outside)
    monkeypatch.setattr("ouroboros.tools.subagent_integration._verify_shared_external_workspace",
                        lambda *a, **k: pytest.fail("escaped source was read"))
    assert "WORKSPACE_MISSING" in invoke(parent)
    assert outside.read_bytes() == b"private sentinel\n"
    assert (art / "workspace.patch").is_file()
    assert_unabsorbed(drive)


def test_capture_under_ancestor_repo_cannot_hide_patch_paths_from_policy(env):
    parent, _, b, drive = env
    art = capture(parent, b, drive)
    ancestor = drive.parent
    git(ancestor, "init", "-q")
    git(ancestor, "config", "user.name", "Fixture")
    git(ancestor, "config", "user.email", "fixture@example.invalid")
    (ancestor / "sentinel.txt").write_bytes(b"ancestor\n")
    git(ancestor, "add", "sentinel.txt")
    git(ancestor, "commit", "-qm", "ancestor")
    manifest_path = art / "workspace_patch.json"
    manifest = json.loads(manifest_path.read_text())
    # Path admission derives from captured patch bytes, not these hints.
    manifest["tracked_changed"] = []
    manifest["untracked_included"] = []
    manifest_path.write_text(json.dumps(manifest))
    parent.task_contract = {"resource_policy": {"protected_artifacts": [{
        "id": "reference", "paths": [str(b / "a.txt")], "deny": ["read_bytes"]}]}}
    before = snapshot(b), (art / "workspace.patch").read_bytes()
    assert "INTEGRATE_TARGET_FORBIDDEN" in invoke(parent)
    assert (snapshot(b), (art / "workspace.patch").read_bytes()) == before
    assert_unabsorbed(drive)


def test_ambient_git_locations_cannot_redirect_b_verification(env, monkeypatch):
    parent, a, b, drive = env
    capture(parent, b, drive)
    before = snapshot(a), snapshot(b)
    monkeypatch.setenv("GIT_DIR", str(a / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(a))
    monkeypatch.setenv("GIT_INDEX_FILE", str(a / ".git" / "index"))
    assert "Verified external_workspace child" in invoke(parent)
    # Observe actual trees outside the deliberately contaminated environment.
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(key)
    assert (snapshot(a), snapshot(b)) == before


def dispositions(drive):
    from ouroboros.delegate_evidence import acceptance_patch_dispositions
    return acceptance_patch_dispositions(drive, "parent").get("rows", [])


@pytest.mark.parametrize("refusal", ["missing_assignment", "conflicting_assignment", "presence_root",
                                     "path_policy", "path_escape", "unparseable_patch"])
def test_early_refusals_keep_verdict_and_custody_without_absorption(env, refusal):
    parent, a, b, drive = env
    art = capture(parent, b, drive)
    args = {}
    if refusal == "missing_assignment":
        row_path = drive / "task_results" / "child.json"
        row = json.loads(row_path.read_text())
        row.pop("workspace_root")
        row["task_constraint"].pop("write_root")
        row_path.write_text(json.dumps(row))
    elif refusal == "conflicting_assignment":
        args["target_root"] = str(a)
    elif refusal == "presence_root":
        ceiling(parent, "active_workspace", ".")
    elif refusal == "path_policy":
        parent.task_contract = {"resource_policy": {"protected_artifacts": [{
            "id": "black-box", "paths": [str(b / "a.txt")], "deny": ["read_bytes"]}]}}
    elif refusal == "path_escape":
        (b.parent / "outside.txt").write_bytes(b"private sentinel\n")
        (b / "a.txt").unlink()
        (b / "a.txt").symlink_to(b.parent / "outside.txt")
    else:
        from hashlib import sha256
        (art / "workspace.patch").write_bytes(b"not a patch\n")
        manifest = json.loads((art / "workspace_patch.json").read_text())
        manifest["sha256"] = sha256(b"not a patch\n").hexdigest()
        (art / "workspace_patch.json").write_text(json.dumps(manifest))
    code, outcome = {
        "missing_assignment": ("TARGET_MISSING", "shared_workspace_missing_target"),
        "conflicting_assignment": ("TARGET_MISMATCH", "shared_workspace_target_mismatch"),
        "presence_root": ("INTEGRATE_TARGET_FORBIDDEN", "shared_workspace_read_refused"),
        "path_policy": ("INTEGRATE_TARGET_FORBIDDEN", "shared_workspace_read_refused"),
        "path_escape": ("WORKSPACE_MISSING", "shared_workspace_missing"),
        "unparseable_patch": ("INTEGRATE_PATCH_UNREADABLE", "shared_workspace_patch_unreadable"),
    }[refusal]
    before = snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()
    result = invoke_result(parent, **args)
    out = result.text
    assert result.status == "blocked" and result.code == "INTEGRATION_BLOCKED"
    assert result.meta["identifier"].endswith(code)
    assert code in out and "Captured result retained" in out, out
    recorded = verdict(drive)
    assert recorded["outcome"] == outcome and recorded["applied"] is False
    custody = dispositions(drive)[-1]
    assert custody["disposition"] == outcome and custody["applied"] is False
    if refusal == "missing_assignment":
        # No known target is manufactured, least of all the parent's folder.
        assert recorded["target_root"] == "" and "target_root" not in custody
    else:
        assert Path(recorded["target_root"]).resolve() == b.resolve()
        assert custody["target_root"] == recorded["target_root"]
    assert (snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()) == before
    assert_unabsorbed(drive)


def presence(parent, root, prefix):
    """Real Presence assembly: the profile selects this tool and one read resource."""
    from ouroboros.presence_authority import build_presence_capability_ceiling, presence_ceiling_payload
    from ouroboros.presence_capabilities import PresenceResourceTarget, PresenceToolTarget
    from tests.test_presence_authority import _resolution
    assembled = build_presence_capability_ceiling(
        skill_name="fixture", skill_content_hash="c" * 64, state_fingerprint="d" * 64,
        resolution=_resolution(PresenceToolTarget("builtin", "integrate_subagent_patch"),
                               PresenceResourceTarget(root, ("read",), prefix)))
    parent.task_contract = {"capability_ceiling": presence_ceiling_payload(assembled)}


@pytest.mark.parametrize("root, prefix, allowed", [
    ("user_files", "Deliverables/b", True),  # B's deepest containing root is deliverables
    ("deliverables", "b", True),
    ("user_files", "Deliverables/other", False),
    ("active_workspace", ".", False),
])
def test_presence_grant_on_overlapping_root_reads_the_same_b(env, monkeypatch, root, prefix, allowed):
    parent, a, _, drive = env
    deliverables = a.parent / "Deliverables"
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(deliverables))
    deliverables.mkdir()
    b = deliverables / "b"
    init(b)
    art = capture(parent, b, drive)
    presence(parent, root, prefix)
    before = snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()
    out = invoke(parent)
    assert (snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()) == before
    if allowed:
        assert "Verified external_workspace child" in out, out
        assert verdict(drive)["target_root"] == str(b.resolve())
        assert load_effective_task_result(drive, "child")["child_result_disposition"] == "integrated"
    else:
        assert "INTEGRATE_TARGET_FORBIDDEN" in out and "Presence" in out, out
        assert_unabsorbed(drive)


@pytest.mark.parametrize("change", ["file_unchanged", "file_to_directory", "directory_to_file"])
def test_registered_output_must_keep_its_recorded_type(env, change):
    parent, a, b, drive = env
    b = b.parent / "plain"
    b.mkdir()
    name = "rendered" if change == "directory_to_file" else "report.txt"
    def edit():
        if change == "directory_to_file":
            (b / name).mkdir()
            (b / name / "frame.bin").write_bytes(b"image\x00")
        else:
            (b / name).write_bytes(b"report\n")
    art = capture(parent, b, drive, direct=True, edit=edit, outputs=(name,))
    if change == "file_to_directory":
        (b / name).unlink()
        (b / name).mkdir()
        (b / name / "inner.txt").write_bytes(b"report\n")
    elif change == "directory_to_file":
        (b / name / "frame.bin").unlink()
        (b / name).rmdir()
        (b / name).write_bytes(b"image\x00")
    before = snapshot(a), snapshot(b)
    out = invoke(parent)
    assert (snapshot(a), snapshot(b)) == before
    assert (art / "workspace_patch.json").is_file()
    if change == "file_unchanged":
        assert "Verified 1 registered file postimage(s)" in out, out
        assert verdict(drive)["outcome"] == "verified_registered_outputs"
        return
    assert "INTEGRATE_DIRECTORY_OUTPUT_MISMATCH" in out, out
    assert verdict(drive)["outcome"] == "direct_output_mismatch"
    assert_unabsorbed(drive)


def test_verdict_target_reaches_acceptance_dispositions_and_absence_stays_absent(env):
    from ouroboros import delegate_custody
    parent, _, b, drive = env
    capture(parent, b, drive)
    assert "Verified external_workspace child" in invoke(parent)
    assert delegate_custody.emit(drive, "delegate_run_patch_verdict", {
        "run_id": "", "task_id": "parent", "child_task_id": "legacy", "pipeline": "subagent",
        "disposition": "verified_shared_workspace", "applied": False, "reason": "",
        "patch_sha256": "", "verdict_artifact_write_failed": False})
    assert "Rejected" in invoke(parent, decision="reject")
    rows = {(row["child"], row["disposition"]): row for row in dispositions(drive)}
    assert rows[("child", "verified_shared_workspace")]["target_root"] == str(b.resolve())
    assert "target_root" not in rows[("legacy", "verified_shared_workspace")]
    assert "target_root" not in rows[("child", "rejected")]


def test_capture_compatible_git_conversion_verifies_crlf_at_b(env, monkeypatch, tmp_path):
    parent, a, b, drive = env
    config = tmp_path / "global.gitconfig"
    config.write_text("[core]\n\tautocrlf = true\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    (b / "crlf.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
    git(b, "add", "crlf.txt")
    git(b, "commit", "-qm", "crlf base")
    art = capture(parent, b, drive, edit=lambda: (b / "crlf.txt").write_bytes(b"one\r\nchanged\r\nthree\r\n"))
    # Capture diffed B through the configured conversion: the patch carries LF lines.
    assert b"+changed\n" in (art / "workspace.patch").read_bytes()
    before = snapshot(a), snapshot(b)
    locations = {"GIT_DIR": a / ".git", "GIT_WORK_TREE": a, "GIT_INDEX_FILE": a / ".git" / "index"}
    for key, value in locations.items():
        monkeypatch.setenv(key, str(value))
    out = invoke(parent)
    for key in locations:
        monkeypatch.delenv(key)
    assert "Verified external_workspace child" in out, out
    assert (snapshot(a), snapshot(b)) == before
    assert (b / "crlf.txt").read_bytes() == b"one\r\nchanged\r\nthree\r\n"


@pytest.mark.parametrize("b_is_room", [False, True])
def test_direct_chat_project_room_parent_verifies_child_folder(env, b_is_room):
    parent, a, b, drive = env
    parent.workspace_root, parent.workspace_mode = None, ""
    parent.is_direct_chat = True
    parent.task_metadata = {"_project_room_dir": str(a)}
    assert parent.active_repo_dir() == a.resolve()
    b = a if b_is_room else b
    art = capture(parent, b, drive)
    before = snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()
    out = invoke(parent)
    assert "Verified external_workspace child" in out, out
    assert verdict(drive)["target_root"] == str(b.resolve())
    assert (snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()) == before
    assert load_effective_task_result(drive, "child")["child_result_disposition"] == "integrated"


@pytest.mark.parametrize("operation", ["read_bytes", "hash"])
@pytest.mark.parametrize("b_is_room", [False, True])
def test_project_room_relative_policy_uses_caller_room(env, monkeypatch, operation, b_is_room):
    parent, a, b, drive = env
    parent.workspace_root, parent.workspace_mode = None, ""
    parent.is_direct_chat = True
    parent.task_metadata = {"_project_room_dir": str(a)}
    b = a if b_is_room else b
    # B's ordinary read binding has B as its base; it must not rebase A's policy.
    monkeypatch.setattr("ouroboros.tool_access._user_files_root", lambda: b)
    if operation == "hash":
        monkeypatch.setattr("ouroboros.workspace_patch_capture._PATCH_FILE_REFERENCE_BYTES", 1)
    art = capture(parent, b, drive)
    if operation == "hash":
        assert json.loads((art / "workspace_patch.json").read_text())["file_output_changes"]
    parent.task_contract = {"resource_policy": {"protected_artifacts": [{
        "id": "room-only", "paths": ["a.txt"], "deny": [operation]}]}}
    registry = ToolRegistry(repo_dir=parent.repo_dir, drive_root=drive)
    registry.set_context(parent)
    read = registry.execute_result("read_file", {"root": "active_workspace", "path": "a.txt"})
    assert ("RESOURCE_POLICY_BLOCKED" in read.text) == (operation == "read_bytes"), read.text
    before = snapshot(a), snapshot(b), snapshot(art)
    result = registry.execute_result("integrate_subagent_patch", {"task_id": "child"})
    if b_is_room:
        assert "RESOURCE_POLICY_BLOCKED" in result.text, result.text
        assert repr(operation) in result.text
        assert result.status == "blocked"
        assert verdict(drive)["outcome"] == "shared_workspace_read_refused"
        assert_unabsorbed(drive)
    else:
        assert "Verified external_workspace child" in result.text, result.text
        assert load_effective_task_result(drive, "child")["child_result_disposition"] == "integrated"
    assert verdict(drive)["target_root"] == str(b.resolve())
    assert verdict(drive)["applied"] is False
    assert (snapshot(a), snapshot(b), snapshot(art)) == before


@pytest.mark.parametrize("room", ["missing_path", "note_only"])
@pytest.mark.parametrize("absolute_policy", [False, True])
def test_unavailable_project_room_does_not_block_independent_child(env, room, absolute_policy):
    parent, a, b, drive = env
    parent.workspace_root, parent.workspace_mode = None, ""
    parent.is_direct_chat = True
    parent.task_metadata = ({"_project_room_dir": str(a.parent / "missing")}
                            if room == "missing_path" else {"_project_room_note": "room unavailable"})
    if absolute_policy:
        parent.task_contract = {"resource_policy": {"protected_artifacts": [{
            "id": "parent-only", "paths": [str(a / "a.txt")], "deny": ["read_bytes", "hash"]}]}}
    art = capture(parent, b, drive)
    before = snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()
    out = invoke(parent)
    assert "Verified external_workspace child" in out, out
    assert verdict(drive)["target_root"] == str(b.resolve())
    assert verdict(drive)["applied"] is False
    assert (snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()) == before
    assert load_effective_task_result(drive, "child")["child_result_disposition"] == "integrated"


def test_inherited_acting_parent_verifies_its_child_at_b(env):
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    parent, a, b, drive = env
    constraint, root, mode, error = _resolve_subagent_constraint(
        SimpleNamespace(REPO_DIR=parent.repo_dir, DRIVE_ROOT=drive), tid="parent",
        requested_constraint={"mode": "acting_subagent", "surface": "external_workspace", "write_root": str(a)},
        workspace_root=str(a), workspace_mode="external", base_sha="", parent_task_id="root")
    assert not error
    parent.task_constraint = normalize_task_constraint(constraint)
    parent.workspace_root, parent.workspace_mode = Path(root), mode
    art = capture(parent, b, drive)
    before = snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()
    out = invoke(parent)
    assert "Verified external_workspace child" in out, out
    assert verdict(drive)["target_root"] == str(b.resolve())
    assert (snapshot(a), snapshot(b), (art / "workspace.patch").read_bytes()) == before
