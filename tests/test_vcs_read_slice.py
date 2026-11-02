"""Local Git read binding, containment, and helper isolation."""

from __future__ import annotations

import json
import os
import pathlib
import shlex
import shutil
import subprocess

import pytest


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def repo(path: pathlib.Path) -> pathlib.Path:
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.name", "Test")
    git(path, "config", "user.email", "test@example.com")
    # Payloads the reads must return byte-exact are LF on every host (a fresh repository converts nothing).
    (path / "note.txt").write_text("first\n", encoding="utf-8", newline="\n")
    git(path, "add", "note.txt")
    git(path, "commit", "-qm", "first")
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system = repo(tmp_path / "system")
    project = repo(tmp_path / "project")
    data = tmp_path / "data"
    data.mkdir()
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=data,
                      workspace_root=project, workspace_mode="external", task_id="read")
    registry = ToolRegistry(system, data)
    registry.set_context(ctx)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    return registry, ctx, system, project


def read(ctx, **args):
    from ouroboros.tools.tool_resolution import _build_builtin_target_binding
    from ouroboros.tools.vcs_read import _vcs_read

    args.setdefault("root", "active_workspace")
    binding = _build_builtin_target_binding(ctx, "vcs_read", args)
    return _vcs_read(ctx, _resolved_binding=binding, **args)


def read_ok(ctx, **args):
    output = read(ctx, **args)
    assert "GIT_ERROR" not in output, (args, output)
    return output


def test_repository_is_selected_before_path_and_nested_repo_is_distinct(world):
    _registry, ctx, _system, project = world
    nested = repo(project / "nested")
    (project / "note.txt").write_text("project dirty\n", encoding="utf-8")
    (nested / "note.txt").write_text("nested dirty\n", encoding="utf-8")
    head = git(nested, "rev-parse", "HEAD")

    from ouroboros.tools.tool_resolution import _build_builtin_target_binding

    selected = _build_builtin_target_binding(ctx, "vcs_read", {
        "root": "active_workspace", "repository": "nested", "path": "note.txt",
    })
    assert selected.base_path == project and selected.repository_path == nested
    assert selected.target_path == nested / "note.txt"
    assert "first" in read_ok(ctx, action="show", repository="nested", path="note.txt")
    assert read_ok(ctx, action="file", repository="nested", path="note.txt", ref=head) == (
        f"first\n\n\nVCS target: root=active_workspace; repo={nested}"
    )
    assert head in read_ok(ctx, action="rev_parse", repository="nested", ref="HEAD")
    blame = read_ok(ctx, action="blame", repository="nested", path="note.txt")
    assert "first" in blame and head[:7] in blame
    assert "project dirty" not in read_ok(ctx, action="file", repository="nested", path="note.txt")
    assert "first" in read_ok(ctx, action="file", path="note.txt")
    with pytest.raises(ValueError):
        _build_builtin_target_binding(ctx, "vcs_read", {"repository": "nested", "path": "../note.txt"})
    with pytest.raises(ValueError):
        _build_builtin_target_binding(ctx, "vcs_read", {"repository": "nested/note.txt"})
    with pytest.raises(ValueError):
        _build_builtin_target_binding(ctx, "vcs_read", {"root": "system_repo", "repository": str(nested)})


def test_literal_metacharacter_filename_blame_and_file_whitespace(world):
    _registry, ctx, _system, project = world
    # Glob, shell and history metacharacters; NTFS forbids the glob ``?`` in a name, so Windows checks the rest.
    filename = "literal [x] $cash &!.txt" if os.name == "nt" else "literal [x] $cash ?&!.txt"
    content = "alpha  \n\nlast\t \n\n"
    (project / filename).write_text(content, encoding="utf-8", newline="\n")
    git(project, "add", "--", filename)
    git(project, "commit", "-qm", "literal filename")
    head = git(project, "rev-parse", "HEAD")

    blame = read_ok(ctx, action="blame", path=filename, ref=head)
    assert "alpha  " in blame and "last" in blame and head[:7] in blame
    assert filename in read_ok(ctx, action="show", path=filename, ref=head)
    assert "literal filename" in read_ok(ctx, action="log", path=filename)
    assert read_ok(ctx, action="file", path=filename, ref=head) == (
        content + f"\n\nVCS target: root=active_workspace; repo={project}"
    )
    (project / "empty.txt").write_text("", encoding="utf-8")
    git(project, "add", "empty.txt")
    git(project, "commit", "-qm", "empty file")
    assert read_ok(ctx, action="file", path="empty.txt") == (
        f"\n\nVCS target: root=active_workspace; repo={project}"
    )
    for absent in ("*.txt", "literal [y] ?.txt"):  # a pattern-looking name is never a pathspec match
        assert "GIT_ERROR" in read(ctx, action="file", path=absent)


def test_identity_unborn_dirty_refs_merge_base_and_linked_worktree(world):
    _registry, ctx, _system, project = world
    first = git(project, "rev-parse", "HEAD")
    (project / "note.txt").write_text("second\n", encoding="utf-8")
    git(project, "add", "note.txt")
    git(project, "commit", "-qm", "second")
    second = git(project, "rev-parse", "HEAD")
    (project / "dirty.txt").write_text("untracked", encoding="utf-8")
    identity = json.loads(read_ok(ctx, action="identity").split("\n\nVCS target:")[0])
    assert identity["head"] == second and identity["dirty"] is True
    assert pathlib.Path(identity["worktree"]).resolve() == project.resolve()  # Git spells it with "/"
    assert first in read_ok(ctx, action="merge_base", base=first, head=second)
    assert "refs/heads/" in read_ok(ctx, action="refs")
    assert "first" in read_ok(ctx, action="log", max_count=2)
    assert "GIT_ERROR" in read(ctx, action="rev_parse", ref="absent-ref")

    linked = project / "linked"
    git(project, "worktree", "add", "-q", "-b", "linked-test", str(linked), second)
    linked_identity = json.loads(read_ok(ctx, action="identity", repository="linked").split("\n\nVCS target:")[0])
    assert pathlib.Path(linked_identity["worktree"]).resolve() == linked.resolve()
    assert linked_identity["git_dir"] != linked_identity["common_dir"]
    assert linked_identity["head"] == second

    # Direct binding uses the same prepared root; a nested unborn repository
    # under the active workspace is enough to exercise the no-HEAD case.
    unborn_nested = project / "unborn"
    unborn_nested.mkdir()
    git(unborn_nested, "init", "-q")
    state = json.loads(read_ok(ctx, action="identity", repository="unborn").split("\n\nVCS target:")[0])
    assert state["head"] is None and state["branch"]

    detached = repo(project / "detached")
    git(detached, "checkout", "--detach", "-q")
    detached_state = json.loads(read_ok(ctx, action="identity", repository="detached").split("\n\nVCS target:")[0])
    assert detached_state["head"] == git(detached, "rev-parse", "HEAD")
    assert detached_state["branch"] is None

    corrupt = repo(project / "corrupt")
    current_ref = git(corrupt, "symbolic-ref", "HEAD")
    (corrupt / ".git" / current_ref).write_text("invalid-object-id\n", encoding="utf-8")
    assert "GIT_ERROR" in read(ctx, action="identity", repository="corrupt")


def test_identity_dirty_uses_the_same_git_config_as_status(world, tmp_path, monkeypatch):
    _registry, ctx, _system, project = world
    ignore = tmp_path / "global-ignore"
    ignore.write_text("*.local\n", encoding="utf-8")
    config = tmp_path / "global.gitconfig"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    (project / "scratch.local").write_text("ignored by the operator\n", encoding="utf-8")

    def dirty() -> bool:
        return json.loads(read_ok(ctx, action="identity").split("\n\nVCS target:")[0])["dirty"]

    assert dirty() is True and "scratch.local" in read_ok(ctx, action="status")
    config.write_text(f"[core]\n\texcludesFile = {ignore.as_posix()}\n", encoding="utf-8")
    assert dirty() is False and "scratch.local" not in read_ok(ctx, action="status")


def test_explicit_repository_status_diff_and_omission_parity(world):
    registry, ctx, _system, project = world
    nested = repo(project / "nested")
    (project / "note.txt").write_text("outer\n", encoding="utf-8")
    (nested / "note.txt").write_text("inner\n", encoding="utf-8")
    for action in ("status", "diff"):
        output = registry.execute("vcs_read", {"action": action})
        assert "GIT_ERROR" not in output and f"repo={project}" in output
        if action == "diff":
            assert "outer" in output and "inner" not in output
        selected = registry.execute("vcs_read", {"action": action, "repository": "nested", "path": "note.txt"})
        assert "GIT_ERROR" not in selected and f"repo={nested}" in selected
        assert ("note.txt" in selected if action == "status" else "inner" in selected)
        assert "outer" not in selected


def test_historical_protected_file_blocks_file_blame_show_and_diff(world):
    _registry, ctx, _system, project = world
    secret = project / "private" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text("historical secret\n", encoding="utf-8")
    git(project, "add", "private/secret.txt")
    git(project, "commit", "-qm", "secret")
    old = git(project, "rev-parse", "HEAD")
    git(project, "rm", "-q", "private/secret.txt")
    git(project, "commit", "-qm", "remove")
    record = {"id": "secret", "role": "black_box_reference", "paths": [str(secret)],
              "deny": ["read_bytes", "static_introspection"]}
    policy = {"resource_policy": {"protected_artifacts": [record]}}
    ctx.task_contract = policy
    ctx.task_metadata = {"task_contract": policy}
    for action in ("file", "blame", "show"):
        output = read_ok(ctx, action=action, path="private/secret.txt", ref=old)
        assert "RESOURCE_POLICY_BLOCKED" in output and "historical secret" not in output
    assert "RESOURCE_POLICY_BLOCKED" in read_ok(ctx, action="show", path="private", ref=old)
    assert "RESOURCE_POLICY_BLOCKED" in read_ok(ctx, action="show", ref=old)
    assert "historical secret" not in read_ok(ctx, action="log")
    from ouroboros.tools.git_vcs_ops import _git_diff
    from ouroboros.tools.tool_resolution import _build_builtin_target_binding

    args = {"repository": ".", "path": "private", "base": old, "head": "HEAD"}
    binding = _build_builtin_target_binding(ctx, "vcs_read", {**args, "action": "diff"})
    diff = _git_diff(ctx, root="active_workspace", _resolved_binding=binding, **args)
    assert "RESOURCE_POLICY_BLOCKED" in diff and "historical secret" not in diff and "GIT_ERROR" not in diff


def test_hostile_git_env_and_repo_helpers_cannot_execute_or_write(world, monkeypatch):
    _registry, ctx, _system, project = world
    marker = project / "helper-ran"
    helper = project / "helper.sh"
    helper.write_text(f"#!/bin/sh\ntouch {shlex.quote(marker.as_posix())}\nprintf hijacked\n",
                      encoding="utf-8", newline="\n")
    helper.chmod(0o755)
    # Git hands these values to sh, which would eat a Windows path's backslashes: then nothing could
    # execute and the refusals below would hold vacuously.
    command = shlex.quote(helper.as_posix())
    (project / ".gitattributes").write_text("*.txt diff=spy\n", encoding="utf-8")
    git(project, "add", ".gitattributes")
    git(project, "commit", "-qm", "attributes")
    (project / "note.txt").write_text("second\n", encoding="utf-8", newline="\n")
    git(project, "add", "note.txt")
    git(project, "commit", "-qm", "second")
    git(project, "config", "diff.spy.textconv", command)
    # Prove the configured converter can execute for this exact file/commit.
    git(project, "show", "--textconv", "--no-ext-diff", "HEAD", "--", "note.txt")
    assert marker.exists()
    marker.unlink()
    git(project, "config", "diff.external", command)
    git(project, "config", "core.pager", command)
    git(project, "config", "core.fsmonitor", command)
    (project / "note.txt").write_text("changed\n", encoding="utf-8")
    monkeypatch.setenv("GIT_DIR", str(project / "missing-git-dir"))
    monkeypatch.setenv("GIT_PAGER", command)
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", command)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "diff.external")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", command)
    index = project / ".git" / "index"
    before_index = index.read_bytes()
    before_objects = {p.relative_to(project / ".git" / "objects")
                      for p in (project / ".git" / "objects").rglob("*") if p.is_file()}
    assert "second" in read_ok(ctx, action="show", path="note.txt")
    assert "second\n" in read_ok(ctx, action="file", path="note.txt")
    assert json.loads(read_ok(ctx, action="identity").split("\n\nVCS target:")[0])["dirty"] is True
    assert not marker.exists()
    assert index.read_bytes() == before_index
    assert before_objects == {p.relative_to(project / ".git" / "objects")
                              for p in (project / ".git" / "objects").rglob("*") if p.is_file()}


def test_registered_reader_and_repository_arguments_work_without_shell(world):
    from ouroboros.contracts.task_constraint import TaskConstraint

    registry, ctx, _system, project = world
    nested = repo(project / "nested-public")
    (nested / "note.txt").write_text("public nested change\n", encoding="utf-8")
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    registry.set_context(ctx)
    assert registry.get_schema_by_name("run_command") is None
    assert registry.get_schema_by_name("vcs_read") is not None
    for action, extra, expected in [
        ("file", {"path": "note.txt"}, "first\n"),
        ("blame", {"path": "note.txt"}, "first"),
        ("log", {}, "first"),
        ("identity", {}, str(nested)),
    ]:
        answer = registry.execute_result("vcs_read", {"action": action, "repository": "nested-public", **extra})
        assert answer.status == "ok" and answer.code != "GIT_ERROR", answer.text
        assert expected in answer.text and f"repo={nested}" in answer.text
    for action in ("status", "diff"):
        answer = registry.execute_result("vcs_read", {"action": action, "repository": "nested-public", "path": "note.txt"})
        assert answer.status == "ok" and answer.code != "GIT_ERROR", answer.text
        assert "note.txt" in answer.text and f"repo={nested}" in answer.text
    denied = registry.execute_result("vcs_read", {"action": "file", "repository": "../system", "path": "note.txt"})
    assert "first\n" not in denied.text and (denied.status != "ok" or denied.code == "GIT_ERROR")


def test_historical_symlink_uses_tree_entry_not_current_target(world):
    registry, ctx, _system, project = world
    link = project / "link.txt"
    try:
        link.symlink_to("note.txt")
    except OSError:
        pytest.skip("host cannot create symlinks")
    git(project, "add", "link.txt")
    git(project, "commit", "-qm", "link")
    result = registry.execute("vcs_read", {"action": "file", "path": "link.txt"})
    assert result.startswith("note.txt\n\nVCS target:"), result
    (project / "second.txt").write_text("second\n", encoding="utf-8")
    git(project, "add", "second.txt")
    git(project, "commit", "-qm", "second")
    link.unlink()
    link.symlink_to("second.txt")
    for spelling in ("link.txt", str(link)):  # an absolute spelling names the same tree entry
        historical = registry.execute("vcs_read", {"action": "file", "path": spelling})
        assert historical.startswith("note.txt\n\nVCS target:"), (spelling, historical)
    status = registry.execute("vcs_read", {"action": "status", "repository": ".", "path": "link.txt"})
    assert "link.txt" in status and "second.txt" not in status, status
    difference = registry.execute("vcs_read", {"action": "diff", "repository": ".", "path": "link.txt"})
    assert "a/link.txt" in difference and "+second.txt" in difference, difference


def test_registered_nested_repositories_keep_presence_and_policy_root(world):
    from ouroboros.presence_authority import (
        PresenceCapabilityCeiling,
        PresenceResourceGrant,
        PresenceToolGrant,
        presence_ceiling_payload,
    )

    registry, ctx, _system, project = world
    allowed = project / "allowed"
    allowed.mkdir()
    correct = repo(allowed / "nested")
    other = project / "other"
    other.mkdir()
    wrong = repo(other / "nested")
    (wrong / "allowed").mkdir()
    (wrong / "allowed" / "secret.txt").write_text("escaped secret\n", encoding="utf-8")
    git(wrong, "add", "allowed/secret.txt")
    git(wrong, "commit", "-qm", "secret")
    ceiling = PresenceCapabilityCeiling(
        skill_name="reader", skill_content_hash="a" * 64,
        profile_fingerprint="b" * 64, state_fingerprint="c" * 64,
        selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=5,
        tool_grants=tuple(PresenceToolGrant(name) for name in ("vcs_read",)),
        resource_grants=(PresenceResourceGrant("active_workspace", ("vcs",), "allowed"),),
        digest="0" * 64,
    )
    ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
    registry.set_context(ctx)
    for name, args in (
        ("vcs_read", {"root": "active_workspace", "action": "file", "repository": "other/nested", "path": "allowed/secret.txt"}),
        ("vcs_read", {"action": "status", "root": "active_workspace", "repository": "other/nested", "path": "allowed/secret.txt"}),
        ("vcs_read", {"action": "diff", "root": "active_workspace", "repository": "other/nested", "path": "allowed/secret.txt"}),
    ):
        denied = registry.execute_result(name, args)
        assert denied.status == "blocked" and "PRESENCE_RESOURCE_BLOCKED" in denied.text, (name, denied)
        accepted = registry.execute_result(name, {**args, "repository": "allowed/nested", "path": "note.txt"})
        assert accepted.status == "ok" and accepted.code != "PRESENCE_RESOURCE_BLOCKED", (name, accepted)
        assert f"repo={correct}" in accepted.text
    assert "first\n" in registry.execute_result("vcs_read", {
        "action": "file", "repository": "allowed/nested", "path": "note.txt",
    }).text


def test_registered_historical_deleted_directory_and_relative_policy(world):
    registry, ctx, _system, project = world
    nested = repo(project / "nested")
    private = nested / "private"
    private.mkdir()
    (private / "secret.txt").write_text("historical directory secret\n", encoding="utf-8")
    git(nested, "add", "private/secret.txt")
    git(nested, "commit", "-qm", "secret")
    old = git(nested, "rev-parse", "HEAD")
    git(nested, "rm", "-q", "private/secret.txt")
    git(nested, "commit", "-qm", "remove")
    assert not private.exists()
    policy = {"resource_policy": {"protected_artifacts": [{
        "id": "private", "role": "black_box_reference", "paths": ["nested/private"],
        "deny": ["read_bytes", "static_introspection"],
    }]}}
    ctx.task_contract = policy
    ctx.task_metadata = {"task_contract": policy}
    registry.set_context(ctx)
    for action, extra in (
        ("file", {"path": "private/secret.txt", "ref": old}),
        ("blame", {"path": "private/secret.txt", "ref": old}),
        ("show", {"path": "private/secret.txt", "ref": old}),
        ("show", {"path": "private", "ref": old}),
        ("show", {"ref": old}),
    ):
        result = registry.execute_result("vcs_read", {"action": action, "repository": "nested", **extra})
        assert result.code == "RESOURCE_POLICY_BLOCKED", (action, result)
        assert "historical directory secret" not in result.text
    difference = registry.execute_result("vcs_read", {"action": "diff",
        "repository": "nested", "path": "private", "base": old, "head": "HEAD",
    })
    assert "RESOURCE_POLICY_BLOCKED" in difference.text and "historical directory secret" not in difference.text
    public = registry.execute_result("vcs_read", {"action": "file", "repository": "nested", "path": "note.txt"})
    assert public.status == "ok" and "first\n" in public.text


def test_registered_protected_ancestor_blocks_unfiltered_history_without_sibling_spill(world):
    registry, ctx, _system, project = world
    nested = project / "nested"
    nested.mkdir()
    selected = repo(nested / "subrepo")
    sibling = repo(project / "sibling")
    (selected / "note.txt").write_text("selected change\n", encoding="utf-8")
    (sibling / "note.txt").write_text("sibling change\n", encoding="utf-8")

    def set_policy(path, *, allow=None, deny=None):
        record = {"id": "ancestor", "role": "black_box_reference", "paths": [path]}
        if allow is not None:
            record["allow"] = allow
        if deny is not None:
            record["deny"] = deny
        policy = {"resource_policy": {"protected_artifacts": [record]}}
        ctx.task_contract = policy
        ctx.task_metadata = {"task_contract": policy}
        registry.set_context(ctx)

    def history(repository):
        return (
            registry.execute_result("vcs_read", {"action": "show", "repository": repository}),
            registry.execute_result("vcs_read", {"action": "diff", "repository": repository}),
        )

    for protected_path in ("nested", "."):
        set_policy(protected_path, deny=["static_introspection"])
        for result in history("nested/subrepo"):
            assert "RESOURCE_POLICY_BLOCKED" in result.text, result
            assert "selected change" not in result.text
        if protected_path == "nested":
            shown, difference = history("sibling")
            assert shown.status == "ok" and shown.code == "OK", shown
            assert difference.status == "ok" and "sibling change" in difference.text, difference

    set_policy("nested", allow=["static_introspection"])
    shown, difference = history("nested/subrepo")
    assert shown.status == "ok" and shown.code == "OK" and "first" in shown.text, shown
    assert difference.status == "ok" and "selected change" in difference.text, difference
    file_result = registry.execute_result("vcs_read", {
        "action": "file", "repository": "nested/subrepo", "path": "note.txt",
    })
    assert file_result.code == "RESOURCE_POLICY_BLOCKED", file_result


def test_registered_file_content_that_looks_like_tool_error_is_ok(world):
    registry, _ctx, _system, project = world
    content = "⚠️ TOOL_ERROR literal historical content\nsecond line\n"
    (project / "warning.txt").write_text(content, encoding="utf-8", newline="\n")
    git(project, "add", "warning.txt")
    git(project, "commit", "-qm", "warning file")
    result = registry.execute_result("vcs_read", {"action": "file", "path": "warning.txt"})
    assert result.status == "ok" and result.code == "OK", result
    assert result.text == content + f"\n\nVCS target: root=active_workspace; repo={project}"
    error = registry.execute_result("vcs_read", {"action": "file", "path": "missing.txt"})
    assert error.code == "GIT_ERROR"


def test_registered_worktree_reads_keep_configured_clean_and_process_filters(world):
    registry, _ctx, _system, project = world
    clean_marker = project / "clean-ran"
    process_marker = project / "process-ran"
    clean = project / "clean.sh"
    clean.write_text(f"#!/bin/sh\ntouch {shlex.quote(clean_marker.as_posix())}\ncat\n", encoding="utf-8", newline="\n")
    clean.chmod(0o755)
    process = project / "process.sh"
    process.write_text(f"#!/bin/sh\ntouch {shlex.quote(process_marker.as_posix())}\nexit 1\n", encoding="utf-8",
                       newline="\n")
    process.chmod(0o755)
    (project / "process.txt").write_text("first\n", encoding="utf-8")
    (project / ".gitattributes").write_text(
        "note.txt filter=clean-spy\nprocess.txt filter=process-spy\n", encoding="utf-8",
    )
    git(project, "add", "process.txt", ".gitattributes")
    git(project, "commit", "-qm", "filters")
    git(project, "config", "filter.clean-spy.clean", shlex.quote(clean.as_posix()))  # sh keeps it whole on Windows
    git(project, "config", "filter.process-spy.process", shlex.quote(process.as_posix()))
    (project / "note.txt").write_text("changed\n", encoding="utf-8")
    (project / "process.txt").write_text("changed\n", encoding="utf-8")
    subprocess.run(["git", "diff", "--", "note.txt", "process.txt"], cwd=project, capture_output=True)
    assert clean_marker.exists() and process_marker.exists()
    clean_marker.unlink()
    process_marker.unlink()
    for action in ("status", "diff"):
        expected = subprocess.run(["git", action], cwd=project, capture_output=True)
        clean_marker.unlink(missing_ok=True)
        process_marker.unlink(missing_ok=True)
        result = registry.execute_result("vcs_read", {"action": action, "repository": "."})
        assert (result.code == "GIT_ERROR") == bool(expected.returncode), result
        if expected.returncode:
            assert process_marker.exists()  # configured process failure remains visible


def test_registered_log_and_show_do_not_invoke_signature_verifier(world, tmp_path):
    if shutil.which("ssh-keygen") is None:
        pytest.skip("ssh-keygen is required to create a signed local fixture")
    registry, _ctx, _system, project = world
    key = tmp_path / "test-signing-key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    git(project, "config", "gpg.format", "ssh")
    git(project, "config", "user.signingkey", str(key))
    git(project, "config", "commit.gpgsign", "true")
    (project / "note.txt").write_text("signed\n", encoding="utf-8")
    git(project, "add", "note.txt")
    git(project, "commit", "-qm", "signed")
    marker = project / "verifier-ran"
    verifier = project / "verify.sh"
    verifier.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 1\n", encoding="utf-8")
    verifier.chmod(0o755)
    git(project, "config", "gpg.ssh.program", str(verifier))
    signers = project / "allowed-signers"
    signers.write_text(f"test@example.com {key.with_suffix('.pub').read_text(encoding='utf-8')}", encoding="utf-8")
    git(project, "config", "gpg.ssh.allowedSignersFile", str(signers))
    git(project, "config", "log.showSignature", "true")
    subprocess.run(["git", "log", "-1"], cwd=project, capture_output=True, check=True)
    assert marker.exists()
    marker.unlink()
    for action in ("log", "show"):
        result = registry.execute_result("vcs_read", {"action": action})
        assert result.status == "ok" and result.code != "GIT_ERROR", result
        assert "signed" in result.text
    assert not marker.exists()
