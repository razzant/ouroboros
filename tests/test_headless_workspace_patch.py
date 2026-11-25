"""Workspace patch capture and finalization.

Split verbatim out of ``tests/test_headless_cli.py`` by theme. This module
owns ``build_workspace_patch``/``write_workspace_patch_artifacts``: which
files a patch carries, the unborn/invalid HEAD cases, the sensitive-file
vetoes, and the acting-base-sha manifest contract.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import time

import pytest

from ouroboros.headless import (
    ARTIFACT_STATUS_FAILED,
    ARTIFACT_STATUS_READY_WITH_CHANGES,
    _incidental_lockfile_excludes,
    build_workspace_patch,
    finalize_task_artifacts,
    task_artifacts_dir,
    write_workspace_patch_artifacts,
)
from ouroboros.task_results import write_task_result
from ouroboros.utils import utc_now_iso


from tests._headless_cli_shared import (  # noqa: F401  (autouse fixture applies on import)
    _init_repo_with_file,
    _managed_worker_pool_available,
)
from tests._store_shared import read_result
from tests._workspace_git import commit, git, repository


def test_received_ignored_paths_use_original_basis_and_exactly_one_lane(tmp_path):
    from ouroboros.workspace_file_outputs import file_output_changes

    root = repository(tmp_path / "received")
    (root / ".gitignore").write_text("*.ignored\n")
    (root / "tracked.ignored").write_bytes(b"original\n")
    git(root, "add", "-f", ".gitignore", "tracked.ignored")
    commit(root)
    base = git(root, "rev-parse", "HEAD").decode().strip()
    task = {"metadata": {"workspace_preflight": {"git": {"head": base}}},
            "received_file_outputs": {path: {"run_id": "late"} for path in
                                      ["new.ignored", "tracked.ignored", "gone.ignored"]}}
    (root / "tracked.ignored").write_bytes(b"late\n")
    (root / "new.ignored").write_bytes(b"new\n")
    for index in range(2):
        cap = tmp_path / f"capture-{index}"
        _, manifest = write_workspace_patch_artifacts(root, cap, task=task)
        rows = {row["path"]: row for row in file_output_changes(manifest, cap)}
        assert set(rows) == {"new.ignored", "tracked.ignored"}
        assert rows["new.ignored"]["before"] is None
        assert rows["tracked.ignored"]["before"]["size"] == len(b"original\n")
        assert manifest["patch_size"] == 0
        if index == 0:
            assert git(root, "ls-files", "new.ignored") == b""
            git(root, "add", "-f", "new.ignored")
    (root / "tracked.ignored").write_bytes(b"original\n")
    (root / "new.ignored").unlink()
    _, reverted = write_workspace_patch_artifacts(root, tmp_path / "reverted", task=task)
    assert reverted["status"] == "ready_no_changes" and not reverted.get("file_output_changes")
    (root / "tracked.ignored").unlink()
    (root / "new.ignored").symlink_to("tracked.ignored")
    cap = tmp_path / "deleted-and-link"
    _, manifest = write_workspace_patch_artifacts(root, cap, task=task)
    rows = {row["path"]: row for row in file_output_changes(manifest, cap)}
    assert rows["tracked.ignored"]["after"] is None
    assert rows["new.ignored"]["after"]["kind"] == "symlink" and manifest["patch_size"] == 0


def test_received_paths_with_unavailable_original_base_are_named(tmp_path):
    root = repository(tmp_path / "received")
    task = {"task_constraint": {"mode": "acting_subagent", "surface": "self_worktree", "write_root": str(root),
                                "base_sha": "f" * 40}, "received_file_outputs": {"new.ignored": {"run_id": "late"}}}
    _, manifest = write_workspace_patch_artifacts(root, tmp_path / "capture", task=task)
    assert manifest["status"] == "failed" and manifest["errors"][0]["paths"] == ["new.ignored"]


def test_workspace_patch_includes_tracked_and_untracked_files(tmp_path):
    repo = tmp_path / "repo"
    repository(repo, unborn=True)
    (repo / "tracked.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    (repo / "new.txt").write_text("hello\n", encoding="utf-8")

    patch = build_workspace_patch(repo)

    assert "diff --git a/tracked.txt b/tracked.txt" in patch
    assert "+new" in patch
    assert "diff --git" in patch and "new.txt" in patch


def test_workspace_patch_lockfile_without_manifest_is_incidental_only_with_code_changes():
    assert _incidental_lockfile_excludes(["package-lock.json"]) == set()
    assert _incidental_lockfile_excludes(["package-lock.json", "package.json", "app.js"]) == set()
    assert _incidental_lockfile_excludes(["package-lock.json", "app.js"]) == {"package-lock.json"}
    assert _incidental_lockfile_excludes(["pkg/poetry.lock", "pkg/module.py"]) == {"pkg/poetry.lock"}


@pytest.mark.parametrize("surface", ["self_worktree", "external_workspace"])
@pytest.mark.parametrize("change,expected", [
    ("member", {"pnpm-lock.yaml"}),
    ("added_member", {"pnpm-lock.yaml"}),
    ("received_member", {"pnpm-lock.yaml"}),
    ("new_lockfile", {"pnpm-lock.yaml"}),
    ("removed_member", {"pnpm-lock.yaml"}),
    ("unrelated_package", {"examples/standalone/pnpm-lock.yaml"}),
    ("incidental_churn", set()),
    ("lockfile_only", {"pnpm-lock.yaml"}),
    ("same_directory", {"pnpm-lock.yaml"}),
])
def test_workspace_patch_keeps_only_pnpm_locks_owned_by_changed_manifests(tmp_path, surface, change, expected):

    repo = repository(tmp_path / "repo")
    package = "no/package.json"
    independent = "examples/standalone/package.json"
    sub_lock = "examples/standalone/pnpm-lock.yaml"
    lockfile = "pnpm-lock.yaml"
    settings = "lockfileVersion: '9.0'\n"
    workspace = settings + "importers:\n  .: {}\n"
    member = "  no: {}\n"
    for rel in (package, independent, "package.json", "app.js"):
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel != package or change not in {"added_member", "received_member"}:
            path.write_text("{}\n", encoding="utf-8")
    if change != "new_lockfile":
        (repo / lockfile).write_text(workspace + (member if change != "added_member" else ""), encoding="utf-8")
    # A standalone pnpm project can have its own lock, but the root lock doesn't own it.
    (repo / sub_lock).write_text(settings, encoding="utf-8")
    base = commit(repo)

    (repo / lockfile).write_text(
        workspace + (member if change != "removed_member" else "") + "settings: {}\n", encoding="utf-8")
    changed_manifest = {"member": package, "added_member": package, "new_lockfile": package, "unrelated_package": independent,
                        "same_directory": "package.json"}.get(change)
    if changed_manifest:
        (repo / changed_manifest).write_text('{"dependencies":{"fixture":"1.0.0"}}\n', encoding="utf-8")
    if change == "unrelated_package":
        (repo / sub_lock).write_text(settings + "dependencies: {}\n", encoding="utf-8")
    elif change == "removed_member":
        (repo / package).unlink()
    elif change == "incidental_churn":
        (repo / "app.js").write_text("export default 1;\n", encoding="utf-8")

    constraint = {"mode": "acting_subagent", "surface": surface, "write_root": str(repo), "base_sha": base}
    task = {"task_constraint": constraint}
    if change == "received_member":
        (repo / package).write_text('{"version":"2"}\n', encoding="utf-8")
        (repo / "app.js").write_text("export default 2;\n", encoding="utf-8")
        (repo / sub_lock).write_text(settings + "settings: {}\n", encoding="utf-8")
        task["received_file_outputs"] = {package: {"run_id": "producer"}}
    _artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "capture", task=task)
    assert manifest["status"] == "ready_with_changes", manifest["errors"]
    patch = (tmp_path / "capture" / "workspace.patch").read_text(encoding="utf-8")
    for path in (lockfile, sub_lock):
        assert (f"diff --git a/{path} b/{path}\n" in patch) == (path in expected)
    excluded = {item["path"] for item in manifest["tracked_excluded"] if "incidental lockfile" in item["reason"]}
    if change == "received_member":
        from ouroboros.workspace_file_outputs import file_output_changes

        assert [row["path"] for row in file_output_changes(manifest, tmp_path / "capture")] == [package]
        assert package not in patch and excluded == {sub_lock}
    else:
        assert excluded == ({lockfile} if lockfile not in expected else set())
    assert not any(item.get("reason") == "pnpm_membership_unknown" for item in manifest["diagnostics"])
    if change == "new_lockfile":
        assert lockfile in manifest["untracked_included"]


@pytest.mark.parametrize("version", ["current", "baseline"])
@pytest.mark.parametrize("fault,detail", [
    ("oversize", "metadata limit"),
    ("invalid_yaml", "YAML"),
    ("unreadable", "read"),
])
def test_workspace_patch_retains_lock_when_pnpm_membership_is_unknown(tmp_path, monkeypatch, version, fault, detail):
    from ouroboros import workspace_patch_capture as capture

    root = repository(tmp_path / "repo")
    lock = root / "pnpm-lock.yaml"
    package = root / "client" / "package.json"
    package.parent.mkdir()
    package.write_text("{}\n")
    complete = b"lockfileVersion: '9.0'\nimporters:\n  .: {}\n"
    member = complete + b"  client: {}\n"
    if fault == "oversize":
        member += b"#" + b"x" * capture._LOCKFILE_METADATA_BYTES + b"\n"
    elif fault == "invalid_yaml":
        member += b"broken: [\n"
    lock.write_bytes(member if version == "baseline" else complete)
    base = commit(root)
    lock.write_bytes(member if version == "current" else complete)
    package.write_text('{"version":"2"}\n')

    if fault == "unreadable" and version == "current":
        read = capture.work_domain.read_bytes

        def fail_metadata_read(path, *, limit=None):
            if path == lock and limit == capture._LOCKFILE_METADATA_BYTES + 1:
                raise OSError("could not read current lockfile")
            return read(path, limit=limit)

        monkeypatch.setattr(capture.work_domain, "read_bytes", fail_metadata_read)
    elif fault == "unreadable":
        git_bytes = capture._git_bytes

        def fail_metadata_blob(cmd, cwd, **kwargs):
            if cmd == ["git", "cat-file", "blob", f"{base}:pnpm-lock.yaml"]:
                return b""  # _git_bytes returns empty bytes for a failed read.
            return git_bytes(cmd, cwd, **kwargs)

        monkeypatch.setattr(capture, "_git_bytes", fail_metadata_blob)

    folder = tmp_path / "capture"
    task = {"task_constraint": {"mode": "acting_subagent", "surface": "self_worktree",
                                "write_root": str(root), "base_sha": base}}
    _, manifest = write_workspace_patch_artifacts(root, folder, task=task)
    assert manifest["status"] == "ready_with_changes", manifest["errors"]
    assert not manifest["tracked_excluded"]
    patch = (folder / "workspace.patch").read_bytes()
    assert b"diff --git a/pnpm-lock.yaml b/pnpm-lock.yaml\n" in patch
    assert b"diff --git a/client/package.json b/client/package.json\n" in patch
    notes = [row for row in manifest["diagnostics"] if row.get("reason") == "pnpm_membership_unknown"]
    assert len(notes) == 1 and notes[0]["path"] == lock.name and detail in notes[0]["detail"]


def test_workspace_patch_preserves_lockfile_when_other_changes_are_junk(tmp_path):
    repo = tmp_path / "repo"
    repository(repo, unborn=True)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    # Version 3: generated output is excluded because the PROJECT declares it,
    # not because a host name rule guesses. Without the .gitignore the dist file
    # would ride the patch and, being a real change beside the lockfile, would
    # also stop the lockfile from reading as incidental.
    (repo / ".gitignore").write_text("dist/\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md", ".gitignore"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    (repo / "dist").mkdir()
    (repo / "dist" / "out.txt").write_text("junk\n", encoding="utf-8")

    _artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})
    patch = (tmp_path / "artifacts" / "workspace.patch").read_text(encoding="utf-8")

    assert "package-lock.json" in patch
    assert "dist/out.txt" not in patch
    assert manifest["counts"]["untracked_included"] == 1
    # A git-ignored file is outside the capture universe, so it is not listed
    # as an exclusion either.
    assert manifest["counts"]["untracked_excluded"] == 0


def test_workspace_patch_excludes_binary_junk_and_oversize(tmp_path, monkeypatch):
    """T7 (v6.35.0): the real-usage workspace patch drops untracked build/runtime
    binaries, junk artifacts, and oversize blobs (recorded, not silently lost),
    while keeping real source additions."""
    import ouroboros.workspace_patch_capture as workspace_patch_capture

    repo = tmp_path / "repo"
    repository(repo, unborn=True)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "seed.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=repo, check=True, capture_output=True,
    )
    # Untracked additions: a real source file (keep), a compiled binary (drop),
    # a redis dump (drop), a declared log (keep), and an oversize text file (drop).
    (repo / "fix.py").write_text("def fixed():\n    return 1\n", encoding="utf-8")
    (repo / "app").write_bytes(b"\x7fELF\x00\x01\x02\x03binary\x00blob")  # compiled binary
    (repo / "dump.rdb").write_bytes(b"REDIS\x00\x01")
    (repo / "run.log").write_text("noise\n", encoding="utf-8")
    (repo / "htmlcov").mkdir()
    (repo / "htmlcov" / "index.html").write_text("<html></html>\n", encoding="utf-8")  # top-level coverage junk
    monkeypatch.setattr(workspace_patch_capture, "_PATCH_MAX_UNTRACKED_FILE_BYTES", 100)
    (repo / "big.txt").write_text("x" * 200, encoding="utf-8")  # 200 bytes > cap; small files pass size

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["exclude_rules_version"] == 4
    excluded = {item["path"]: item["reason"] for item in manifest["untracked_excluded"]}
    assert "binary file" in excluded.get("app", "")
    assert "binary file" in excluded.get("dump.rdb", "") or "junk artifact" in excluded.get("dump.rdb", "")
    assert "run.log" in manifest["untracked_included"]
    assert "run.log" not in excluded
    assert "junk artifact" in excluded.get("htmlcov/index.html", "")  # top-level htmlcov excluded
    assert "size cap" in excluded.get("big.txt", "")
    assert "fix.py" in manifest["untracked_included"]
    patch = (tmp_path / "artifacts" / "workspace.patch").read_text(encoding="utf-8")
    assert "fix.py" in patch
    assert "diff --git a/app b/app" not in patch
    assert "dump.rdb" not in patch
    assert "run.log" in patch
    assert "big.txt" not in patch


def test_workspace_patch_supports_unborn_git_worktree(tmp_path):
    repo = tmp_path / "repo"
    repository(repo, unborn=True)
    (repo / "created.txt").write_text("hello\n", encoding="utf-8")

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["base_is_empty_tree"] is True
    assert manifest["base_head"] == "(unborn)"
    assert manifest["current_head"] == "(unborn)"
    assert any(item["kind"] == "workspace_patch" for item in artifacts)
    patch = (tmp_path / "artifacts" / "workspace.patch").read_text(encoding="utf-8")
    assert "created.txt" in patch
    assert "+hello" in patch
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=repo, capture_output=True)
    assert head.returncode != 0


def test_workspace_patch_supports_unborn_sha256_git_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init = subprocess.run(["git", "init", "--object-format=sha256"], cwd=repo, capture_output=True)
    if init.returncode != 0:
        pytest.skip("git does not support sha256 object-format")
    (repo / "created.txt").write_text("hello\n", encoding="utf-8")

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["base_is_empty_tree"] is True
    assert len(manifest["base_ref"]) == 64
    assert any(item["kind"] == "workspace_patch" for item in artifacts)


def test_workspace_patch_allows_external_workspace_first_commit(tmp_path):
    from ouroboros.workspace_preflight import _git_snapshot, summarize_workspace_preflight

    repo = tmp_path / "repo"
    repository(repo, unborn=True)
    admission = summarize_workspace_preflight({"git": _git_snapshot(repo)})
    assert admission["git"]["head_state"] == "unborn"
    (repo / "created.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "created.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "first"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    task = {"metadata": {"workspace_preflight": admission}}

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task=task)

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["errors"] == []
    assert any(item["kind"] == "workspace_patch" for item in artifacts)


def test_workspace_patch_fails_on_invalid_head_not_unborn(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    head_ref = subprocess.run(["git", "symbolic-ref", "--quiet", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    ref_path = repo / ".git" / head_ref
    ref_path.unlink()

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == ARTIFACT_STATUS_FAILED
    assert any(error["type"] == "git_invalid_head" for error in manifest["errors"])
    assert not any(item["kind"] == "workspace_patch" for item in artifacts)


def test_workspace_patch_manifest_excludes_env_cache_dirs(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "new.txt").write_text("hello\n", encoding="utf-8")
    (repo / "node_modules" / "pkg").mkdir(parents=True)
    (repo / "node_modules" / "pkg" / "index.js").write_text("generated\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"

    artifacts, manifest = write_workspace_patch_artifacts(repo, artifact_dir, task={})

    assert manifest["status"] == "ready_with_changes"
    assert "new.txt" in (artifact_dir / "workspace.patch").read_text(encoding="utf-8")
    assert "node_modules" not in (artifact_dir / "workspace.patch").read_text(encoding="utf-8")
    assert manifest["counts"]["untracked_excluded"] == 1
    assert any(item["kind"] == "workspace_patch_manifest" for item in artifacts)


def test_workspace_patch_keeps_project_npmrc_and_tracked_diff(tmp_path):
    """A project npm configuration is ordinary content, not the owner's ~/.npmrc."""
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    (repo / ".npmrc").write_text("registry=https://registry.npmjs.org/\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"

    artifacts, manifest = write_workspace_patch_artifacts(repo, artifact_dir, task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["errors"] == []
    assert manifest["sensitive_blocked"] == []
    assert ".npmrc" in manifest["untracked_included"]
    assert any(item["kind"] == "workspace_patch" for item in artifacts)
    patch_text = (artifact_dir / "workspace.patch").read_text(encoding="utf-8")
    assert "tracked.txt" in patch_text
    assert "registry=https://registry.npmjs.org/" in patch_text


def test_workspace_patch_keeps_public_pem_because_a_suffix_is_not_key_material(tmp_path):
    """A public certificate is ordinary work: the suffix no longer refuses it,
    and nothing is reported as sensitive. Real key material is caught by the
    content check pinned in the next test (owner answer 3=A)."""
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    (repo / "public.pem").write_text("-----BEGIN PUBLIC KEY-----\nAAAA\n-----END PUBLIC KEY-----\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"

    artifacts, manifest = write_workspace_patch_artifacts(repo, artifact_dir, task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["sensitive_blocked"] == []
    assert any(item["kind"] == "workspace_patch" for item in artifacts)
    patch_text = (artifact_dir / "workspace.patch").read_text(encoding="utf-8")
    assert "tracked.txt" in patch_text
    assert "public.pem" in patch_text


def test_workspace_patch_excludes_private_key_material_by_content(tmp_path):
    """An innocently NAMED file whose bytes carry a PEM private-key header is
    excluded on content evidence; the tracked diff still survives (#447)."""
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    (repo / "notes.txt").write_text(
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----\n",
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"

    artifacts, manifest = write_workspace_patch_artifacts(repo, artifact_dir, task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert {item["path"]: item["reason"] for item in manifest["untracked_excluded"]} == {
        "notes.txt": "private key material (PEM private-key header)"
    }
    assert any(item["kind"] == "workspace_patch" for item in artifacts)
    patch_text = (artifact_dir / "workspace.patch").read_text(encoding="utf-8")
    assert "tracked.txt" in patch_text
    assert "PRIVATE KEY" not in patch_text


def test_workspace_patch_excludes_dependency_cache_without_inventing_secret_evidence(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    secret = repo / "node_modules" / "pkg" / "service-account.json"
    secret.parent.mkdir(parents=True)
    secret.write_text("TOKEN=secret\n", encoding="utf-8")

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["counts"]["sensitive_blocked"] == 0
    assert manifest["untracked_excluded"] == [{
        "path": "node_modules/pkg/service-account.json",
        "reason": "env/cache directory segment: node_modules",
    }]
    assert any(item["kind"] == "workspace_patch" for item in artifacts)


def test_workspace_patch_keeps_project_credential_names_and_excludes_dotenv(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    (repo / "credentials").write_text("secret\n", encoding="utf-8")
    (repo / "prod.env").write_text("SECRET=1\n", encoding="utf-8")
    (repo / "settings.env.local").write_text("SECRET=1\n", encoding="utf-8")
    (repo / ".aws").mkdir()
    (repo / ".aws" / "credentials").write_text("secret\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"

    artifacts, manifest = write_workspace_patch_artifacts(repo, artifact_dir, task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert {item["path"] for item in manifest["sensitive_blocked"]} == {
        "prod.env",
        "settings.env.local",
    }
    assert {"credentials", ".aws/credentials"} <= set(manifest["untracked_included"])
    assert any(item["kind"] == "workspace_patch" for item in artifacts)
    patch_text = (artifact_dir / "workspace.patch").read_text(encoding="utf-8")
    assert "SECRET" not in patch_text
    assert "a/credentials b/credentials" in patch_text
    assert ".aws/credentials" in patch_text


def test_workspace_patch_allows_benign_tokenizer_json(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["sensitive_blocked"] == []
    assert any(item["kind"] == "workspace_patch" for item in artifacts)


def test_failed_refinalization_drops_stale_workspace_patch_metadata(tmp_path):
    parent = tmp_path / "data"
    repo = tmp_path / "repo"
    parent.mkdir()
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    task = {"id": "task-stale", "workspace_root": str(repo)}
    write_task_result(parent, "task-stale", "completed", workspace_root=str(repo), artifact_status="finalizing")
    finalize_task_artifacts(parent, task)
    result = read_result(parent, "task-stale")
    assert any(item.get("kind") == "workspace_patch" for item in result["artifacts"])

    # A sensitive-shaped file no longer fails the patch (#447); break the HEAD
    # ref instead to force a genuinely FAILED refinalization.
    head_ref = subprocess.run(["git", "symbolic-ref", "--quiet", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / ".git" / head_ref).unlink()
    finalize_task_artifacts(parent, task)

    result = read_result(parent, "task-stale")
    assert result["artifact_status"] == ARTIFACT_STATUS_FAILED
    assert not any(item.get("kind") == "workspace_patch" for item in result["artifacts"])


def test_workspace_patch_preserves_untracked_paths_with_whitespace(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    leading = repo / " leading.txt"
    nested = repo / "dir with space" / "file name.txt"
    leading.write_text("leading\n", encoding="utf-8")
    nested.parent.mkdir()
    nested.write_text("nested\n", encoding="utf-8")

    _artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == "ready_with_changes"
    assert " leading.txt" in manifest["untracked_included"]
    assert "dir with space/file name.txt" in manifest["untracked_included"]
    assert manifest["patch_size"] > 0


def test_finalize_workspace_patch_allows_external_workspace_head_changed(tmp_path):
    parent = tmp_path / "data"
    repo = tmp_path / "repo"
    parent.mkdir()
    _init_repo_with_file(repo)
    old_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "move"], cwd=repo, check=True, capture_output=True)
    task = {
        "id": "task-head",
        "workspace_root": str(repo),
        "metadata": {"workspace_preflight": {"git": {"head": old_head}}},
    }
    write_task_result(parent, "task-head", "completed", workspace_root=str(repo), artifact_status="finalizing")

    finalize_task_artifacts(parent, task)

    result = read_result(parent, "task-head")
    assert result["artifact_status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    manifest = json.loads((task_artifacts_dir(parent, "task-head") / "workspace_patch.json").read_text(encoding="utf-8"))
    assert manifest["errors"] == []


def test_finalize_workspace_patch_exception_manifest_keeps_base_fields(tmp_path, monkeypatch):
    import ouroboros.headless as headless

    parent = tmp_path / "data"
    repo = tmp_path / "repo"
    parent.mkdir()
    _init_repo_with_file(repo)
    task = {"id": "task-exception", "workspace_root": str(repo)}
    write_task_result(parent, "task-exception", "completed", workspace_root=str(repo), artifact_status="finalizing")

    def boom(*_args, **_kwargs):
        raise RuntimeError("artifact failure")

    monkeypatch.setattr(headless, "write_workspace_patch_artifacts", boom)
    headless.finalize_task_artifacts(parent, task)

    result = read_result(parent, "task-exception")
    manifest = json.loads((task_artifacts_dir(parent, "task-exception") / "workspace_patch.json").read_text(encoding="utf-8"))
    assert result["artifact_status"] == ARTIFACT_STATUS_FAILED
    assert manifest["status"] == ARTIFACT_STATUS_FAILED
    assert manifest["base_ref"] == ""
    assert manifest["base_head"] == ""
    assert manifest["base_is_empty_tree"] is False
    assert manifest["current_head"] == ""


def test_workspace_patch_uses_acting_base_sha_without_preflight_metadata(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    base_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / "tracked.txt").write_text("acting edit\n", encoding="utf-8")
    task = {
        "task_constraint": {
            "mode": "acting_subagent",
            "surface": "self_worktree",
            "base_sha": base_head,
        },
    }

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task=task)

    assert manifest["status"] == "ready_with_changes"
    assert manifest["base_ref"] == base_head
    assert manifest["base_head"] == base_head
    assert manifest["current_head"] == base_head
    assert any(item["kind"] == "workspace_patch" for item in artifacts)


def test_workspace_patch_fails_when_acting_base_sha_head_changed(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    base_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / "tracked.txt").write_text("committed by child\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "child commit"], cwd=repo, check=True, capture_output=True)
    moved_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    task = {
        "task_constraint": {
            "mode": "acting_subagent",
            "surface": "self_worktree",
            "base_sha": base_head,
        },
    }

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task=task)

    assert manifest["status"] == ARTIFACT_STATUS_FAILED
    assert manifest["base_ref"] == base_head
    assert manifest["errors"][-1]["type"] == "workspace_head_changed"
    assert manifest["errors"][-1]["expected_head"] == base_head
    assert manifest["errors"][-1]["current_head"] == moved_head
    assert not any(item["kind"] == "workspace_patch" for item in artifacts)


def test_an_untracked_symlink_loop_never_fails_the_capture(tmp_path):
    """A link left pointing at itself (a test's ``snapshot.json -> snapshot.json``) cannot be
    resolved. Only a declared scratch path is resolved, and an unresolvable link is no scratch
    file, so the capture goes on and Git keeps the link as a link."""
    import os

    from ouroboros.workspace_patch_capture import SCRATCH_MANIFEST_NAME

    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "kept.txt").write_text("kept\n", encoding="utf-8")
    for folder in (repo, repo / ".review-evidence"):
        folder.mkdir(exist_ok=True)
        os.symlink("snapshot.json", folder / "snapshot.json")
    for name, scratch in (("plain", None), ("scratch", {str(tmp_path / "elsewhere.txt"): "0" * 64})):
        artifacts = tmp_path / name
        if scratch is not None:
            artifacts.mkdir()
            (artifacts / SCRATCH_MANIFEST_NAME).write_text(json.dumps({"scratch": scratch}), encoding="utf-8")
        _, manifest = write_workspace_patch_artifacts(repo, artifacts, task={})
        assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES, (name, manifest.get("errors"))
        patch = (artifacts / "workspace.patch").read_text(encoding="utf-8")
        assert "kept.txt" in patch and "new file mode 120000" in patch, name


# --- #2711: the copy's evidence folder is held out of every capture by one name and retained once ---

def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _evidence_copy(tmp_path, *, tracked_evidence=False):
    """A task's own copy holding a product change and a session's test runs under ``.review-evidence/``."""
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    if tracked_evidence:
        (repo / ".review-evidence").mkdir()
        (repo / ".review-evidence" / "keep.txt").write_text("the project's own file\n", encoding="utf-8")
        _git(repo, "add", ".review-evidence")
        _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "tracked evidence")
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")
    run = repo / ".review-evidence" / "safe-test-evidence" / "run-1"
    run.mkdir(parents=True)
    expected = {".review-evidence/safe-test-evidence/run-1/result.json": b'{"returncode": 0}\r\n',
                ".review-evidence/safe-test-evidence/run-1/trace.bin": b"\x7fELF\x00\x01" + bytes(range(256)) * 8}
    for name, payload in expected.items():
        (repo / name).write_bytes(payload)
    task = {"id": "task-evidence", "workspace_root": str(repo), "metadata": {"workspace_preflight": {"git": {"head": head}}}}
    return repo, task, expected


def _retained_files(summary_path):
    root = pathlib.Path(summary_path).parent / "files"
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_the_copys_evidence_folder_is_retained_once_and_held_out_of_the_patch_and_file_outputs(tmp_path):
    """#2711: at task finalization the session's test runs under ``.review-evidence/`` reach the task record once,
    as retained evidence files, never again inside ``workspace.patch`` (text) and a file-outputs zip (binaries);
    a second finalization of the unchanged copy copies nothing again."""
    from ouroboros.workspace_patch_rules import NAMED_EVIDENCE_EXCLUSION
    from ouroboros.workspace_patch_capture import _held_named_evidence

    parent = tmp_path / "data"
    parent.mkdir()
    repo, task, expected = _evidence_copy(tmp_path)
    task["metadata"]["host_evidence"] = "old-evidence"  # stale run metadata cannot rename task evidence
    delegated = "nested/.review-evidence/run/log"
    held, summary = _held_named_evidence(repo, "nested/.review-evidence", set(expected) | {delegated}, set(), None, None)
    assert held == {delegated} and summary is None  # a delegated run still uses its own folder
    write_task_result(parent, task["id"], "completed", workspace_root=str(repo), artifact_status="finalizing")
    finalize_task_artifacts(parent, task)
    result = read_result(parent, task["id"])
    assert result["artifact_status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    kinds = [row["kind"] for row in result["artifacts"]]
    assert kinds.count("named_evidence_manifest") == 1 and "workspace_file_outputs" not in kinds, kinds
    base = task_artifacts_dir(parent, task["id"])
    manifest = json.loads((base / "workspace_patch.json").read_text(encoding="utf-8"))
    assert {row["path"]: row["reason"] for row in manifest["untracked_excluded"]} == dict.fromkeys(expected, NAMED_EVIDENCE_EXCLUSION)
    assert manifest["untracked_included"] == ["feature.py"] and manifest["file_outputs"] == []
    patch = (base / "workspace.patch").read_text(encoding="utf-8")
    assert "feature.py" in patch and ".review-evidence" not in patch
    evidence = manifest["named_evidence"]
    assert (evidence["status"], evidence["files"], evidence["gaps"]) == ("retained", 2, 0)
    assert _retained_files(evidence["summary_path"]) == expected
    [row] = [row for row in result["artifacts"] if row["kind"] == "named_evidence_manifest"]
    assert row["path"] == evidence["summary_path"] and row["files"] == 2 and row.get("sha256")
    witness = json.loads(pathlib.Path(evidence["summary_path"]).read_text(encoding="utf-8"))["witness"]
    stamps = {path: path.lstat().st_mtime_ns for path in (pathlib.Path(evidence["summary_path"]).parent / "files").rglob("*")}
    finalize_task_artifacts(parent, {**task, **read_result(parent, task["id"])})  # the copy is unchanged
    again = json.loads((base / "workspace_patch.json").read_text(encoding="utf-8"))["named_evidence"]
    assert again["summary_path"] == evidence["summary_path"] and _retained_files(again["summary_path"]) == expected
    assert json.loads(pathlib.Path(again["summary_path"]).read_text(encoding="utf-8"))["witness"] == witness
    assert {path: path.lstat().st_mtime_ns for path in (pathlib.Path(again["summary_path"]).parent / "files").rglob("*")} == stamps
    assert len(list(base.glob("**/named_evidence*"))) == 1  # one copy, never two
    assert [row["kind"] for row in read_result(parent, task["id"])["artifacts"]].count("named_evidence_manifest") == 1



def test_a_baseline_tracked_path_under_the_evidence_name_stays_product_and_is_not_retained(tmp_path):
    """A path the recorded baseline holds there (Git, or a supplied ``file_baseline``) is the project's own:
    edited, it stays in the patch; only a file new since the baseline is evidence (astra R3's reverse test)."""
    from hashlib import sha256

    parent = tmp_path / "data"
    parent.mkdir()
    repo, task, expected = _evidence_copy(tmp_path, tracked_evidence=True)
    (repo / ".review-evidence" / "keep.txt").write_text("the project's own file, edited\n", encoding="utf-8")
    (repo / ".gitignore").write_text("*.csv\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "ignore csv")
    task["metadata"]["workspace_preflight"]["git"]["head"] = _git(repo, "rev-parse", "HEAD")
    before = b"a,b\n1,2\n"
    (repo / ".review-evidence" / "supplied.csv").write_bytes(before)  # a copied input under the evidence name
    supplied = {"kind": "file", "size": len(before), "sha256": sha256(before).hexdigest(), "mode": 0o644}
    task["metadata"]["file_baseline"] = {".review-evidence/supplied.csv": supplied}
    (repo / ".review-evidence" / "supplied.csv").write_bytes(before + b"3,4\n")
    write_task_result(parent, task["id"], "completed", workspace_root=str(repo), artifact_status="finalizing")
    finalize_task_artifacts(parent, task)
    base = task_artifacts_dir(parent, task["id"])
    manifest = json.loads((base / "workspace_patch.json").read_text(encoding="utf-8"))
    assert ".review-evidence/keep.txt" in manifest["tracked_changed"]
    assert {row["path"] for row in manifest["untracked_excluded"]} == set(expected)
    assert [row["path"] for row in manifest["file_output_changes"]] == [".review-evidence/supplied.csv"]
    patch = (base / "workspace.patch").read_text(encoding="utf-8")
    assert "edited" in patch and "safe-test-evidence" not in patch
    assert _retained_files(manifest["named_evidence"]["summary_path"]) == expected  # neither product path is evidence


def test_a_failed_evidence_retention_keeps_the_files_in_the_capture(tmp_path, monkeypatch):
    """Never zero copies: when the folder cannot be retained, its files ride the patch and the outputs as before,
    and the manifest says why."""
    from ouroboros import delegate_named_evidence

    parent = tmp_path / "data"
    parent.mkdir()
    repo, task, expected = _evidence_copy(tmp_path)

    def refused(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(delegate_named_evidence, "retain_named_evidence", refused)
    write_task_result(parent, task["id"], "completed", workspace_root=str(repo), artifact_status="finalizing")
    finalize_task_artifacts(parent, task)
    result = read_result(parent, task["id"])
    base = task_artifacts_dir(parent, task["id"])
    manifest = json.loads((base / "workspace_patch.json").read_text(encoding="utf-8"))
    assert manifest["named_evidence"]["status"] == "capture_failed" and "No space left" in manifest["named_evidence"]["error"]
    assert ".review-evidence/safe-test-evidence/run-1/result.json" in manifest["untracked_included"]
    assert ".review-evidence/safe-test-evidence/run-1/trace.bin" in [row["path"] for row in manifest["file_output_changes"]]
    assert "named_evidence_manifest" not in [row["kind"] for row in result["artifacts"]]
    assert not list(base.glob("**/named_evidence*")) and not list(base.glob("**/.named_evidence*"))


def test_a_direct_folders_evidence_folder_is_retained_once_beside_its_direct_capture(tmp_path):
    """#2711 for an ordinary folder (no Git): its direct capture retains the session's new ``.review-evidence/``
    files once, as a copy's capture does, while a supplied ``file_baseline`` path there stays the folder's own;
    a folder with no new evidence retains nothing, and an unchanged one is not copied again."""
    from hashlib import sha256

    parent, folder = tmp_path / "data", tmp_path / "folder"
    (folder / ".review-evidence").mkdir(parents=True)
    parent.mkdir()
    before = b"a,b\n1,2\n"
    (folder / ".review-evidence" / "supplied.csv").write_bytes(before)  # a copied input under the evidence name
    supplied = {"kind": "file", "size": len(before), "sha256": sha256(before).hexdigest(), "mode": 0o644}
    task = {"id": "task-direct", "workspace_root": str(folder), "metadata": {"file_baseline": {".review-evidence/supplied.csv": supplied}}}
    write_task_result(parent, task["id"], "completed", workspace_root=str(folder), artifact_status="finalizing")
    finalize_task_artifacts(parent, task)
    base = task_artifacts_dir(parent, task["id"])
    manifest = json.loads((base / "workspace_patch.json").read_text(encoding="utf-8"))
    assert manifest["capture_kind"] == "directory_direct" and "named_evidence" not in manifest
    assert "named_evidence_manifest" not in [row["kind"] for row in read_result(parent, task["id"])["artifacts"]]
    expected = {".review-evidence/safe-test-evidence/run-1/result.json": b'{"returncode": 0}\r\n'}
    for name, payload in expected.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(payload)
    finalize_task_artifacts(parent, {**task, **read_result(parent, task["id"])})
    manifest = json.loads((base / "workspace_patch.json").read_text(encoding="utf-8"))
    evidence = manifest["named_evidence"]
    assert manifest["capture_kind"] == "directory_direct" and (evidence["status"], evidence["files"]) == ("retained", 1)
    assert _retained_files(evidence["summary_path"]) == expected  # the supplied path is no evidence
    assert [row["kind"] for row in read_result(parent, task["id"])["artifacts"]].count("named_evidence_manifest") == 1
    assert (folder / ".review-evidence" / "supplied.csv").read_bytes() == before and (folder / next(iter(expected))).is_file()
    finalize_task_artifacts(parent, {**task, **read_result(parent, task["id"])})  # the folder is unchanged
    assert json.loads((base / "workspace_patch.json").read_text(encoding="utf-8"))["named_evidence"]["summary_path"] == evidence["summary_path"]
    assert len(list(base.glob("**/named_evidence*"))) == 1  # one copy, never two
    assert [row["kind"] for row in read_result(parent, task["id"])["artifacts"]].count("named_evidence_manifest") == 1
    time.sleep(0.05)  # a later task on the same folder starts after the first one's files were written
    later = {"id": "task-later", "workspace_root": str(folder), "started_at": utc_now_iso()}
    time.sleep(0.05)
    (folder / ".review-evidence" / "later.txt").write_bytes(b"mine\n")
    write_task_result(parent, later["id"], "completed", workspace_root=str(folder), artifact_status="finalizing")
    finalize_task_artifacts(parent, later)
    mine = json.loads((task_artifacts_dir(parent, later["id"]) / "workspace_patch.json").read_text(encoding="utf-8"))["named_evidence"]
    assert _retained_files(mine["summary_path"]) == {".review-evidence/later.txt": b"mine\n"}, "the first task's evidence stays its own"


def test_machine_patch_ignores_a_projects_textconv_driver(tmp_path):
    """A project may render a binary type for people through ``diff.<driver>.textconv``; the patch
    a parent applies must still carry the bytes, so `git apply --binary` restores the child's file."""
    from ouroboros.workspace_patch_capture import write_workspace_patch_artifacts

    project = tmp_path / "project"
    repository(project, unborn=True)
    table = project / "table.dat"
    before, after = b"\0\x0a\x01abc", b"\0\x0b\x02def"
    table.write_bytes(before)
    (project / ".gitattributes").write_text("*.dat diff=octets\n", encoding="utf-8")
    _git(project, "config", "diff.octets.textconv", "od -An -tx1")
    _git(project, "add", ".")
    _git(project, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    table.write_bytes(after)
    assert "00 0b 02" in " ".join(_git(project, "diff").split()), "the driver is live: a person's diff shows its rendering"
    _, receipt = write_workspace_patch_artifacts(project, tmp_path / "result", task={"id": "child"})
    assert receipt["status"] == "ready_with_changes"
    table.write_bytes(before)
    applied = subprocess.run(["git", "apply", "--binary", str(tmp_path / "result" / "workspace.patch")],
                             cwd=project, capture_output=True, text=True)
    assert applied.returncode == 0 and table.read_bytes() == after, applied.stderr


@pytest.mark.parametrize("activity", ["quiet", "touch", "append"])
def test_public_patch_survives_benign_build_activity(tmp_path, monkeypatch, activity):
    import os
    from ouroboros import config, workspace_patch_capture as capture

    copy, task, _ = _evidence_copy(tmp_path)
    public = copy / "feature.py"
    before = public.stat()
    classify = capture.untracked_binary_verdicts
    monkeypatch.setattr(config, "load_settings", lambda **kw: {"OPENAI_API_KEY": "unused-fixture-value"})

    def during_capture(*args, **kwargs):
        if activity == "touch":
            os.utime(public, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        elif activity == "append":
            with public.open("a", encoding="utf-8") as stream:
                stream.write("# build complete\\n")
        return classify(*args, **kwargs)

    monkeypatch.setattr(capture, "untracked_binary_verdicts", during_capture)
    out = tmp_path / "out"
    capture.write_workspace_patch_artifacts(copy, out, task=task)
    assert b"feature.py" in (out / "workspace.patch").read_bytes()


@pytest.mark.parametrize("directory", ["node_modules", "__pycache__"])
def test_excluded_path_is_never_opened_for_a_product_patch(tmp_path, monkeypatch, directory):
    from ouroboros import config, review_checking, workspace_patch_capture as capture

    copy, task, _ = _evidence_copy(tmp_path)
    path = copy / directory / "generated.data"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"build output")
    monkeypatch.setattr(config, "load_settings", lambda **kw: {"OPENAI_API_KEY": "unused-fixture-value"})
    open_regular = review_checking._open_regular

    def checked_open(candidate):
        assert candidate != path, "excluded build output must never be scanned for credentials"
        return open_regular(candidate)

    monkeypatch.setattr(review_checking, "_open_regular", checked_open)
    out = tmp_path / "out"
    capture.write_workspace_patch_artifacts(copy, out, task=task)
    assert b"feature.py" in (out / "workspace.patch").read_bytes()
