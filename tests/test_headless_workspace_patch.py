"""Workspace patch capture and finalization.

Split verbatim out of ``tests/test_headless_cli.py`` by theme. This module
owns ``build_workspace_patch``/``write_workspace_patch_artifacts``: which
files a patch carries, the unborn/invalid HEAD cases, the sensitive-file
vetoes, and the acting-base-sha manifest contract.
"""
from __future__ import annotations

import json
import subprocess

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


from tests._headless_cli_shared import (  # noqa: F401  (autouse fixture applies on import)
    _init_repo_with_file,
    _managed_worker_pool_available,
)


def test_workspace_patch_includes_tracked_and_untracked_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
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


def test_workspace_patch_preserves_lockfile_when_other_changes_are_junk(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
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
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "seed.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=repo, check=True, capture_output=True,
    )
    # Untracked additions: a real source file (keep), a compiled binary (drop),
    # a redis dump + log junk (drop), and an oversize text file (drop).
    (repo / "fix.py").write_text("def fixed():\n    return 1\n", encoding="utf-8")
    (repo / "app").write_bytes(b"\x7fELF\x00\x01\x02\x03binary\x00blob")  # compiled binary
    (repo / "dump.rdb").write_bytes(b"REDIS\x00\x01")
    (repo / "run.log").write_text("noise\n", encoding="utf-8")
    (repo / "htmlcov").mkdir()
    (repo / "htmlcov" / "index.html").write_text("<html></html>\n", encoding="utf-8")  # top-level coverage junk
    monkeypatch.setattr(workspace_patch_capture, "_PATCH_MAX_UNTRACKED_FILE_BYTES", 100)
    (repo / "big.txt").write_text("x" * 200, encoding="utf-8")  # 200 bytes > cap; small files pass size

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["exclude_rules_version"] == 3
    excluded = {item["path"]: item["reason"] for item in manifest["untracked_excluded"]}
    assert "binary file" in excluded.get("app", "")
    assert "binary file" in excluded.get("dump.rdb", "") or "junk artifact" in excluded.get("dump.rdb", "")
    assert "junk artifact" in excluded.get("run.log", "")
    assert "junk artifact" in excluded.get("htmlcov/index.html", "")  # top-level htmlcov excluded
    assert "size cap" in excluded.get("big.txt", "")
    assert "fix.py" in manifest["untracked_included"]
    patch = (tmp_path / "artifacts" / "workspace.patch").read_text(encoding="utf-8")
    assert "fix.py" in patch
    assert "diff --git a/app b/app" not in patch
    assert "dump.rdb" not in patch
    assert "run.log" not in patch
    assert "big.txt" not in patch


def test_workspace_patch_supports_unborn_git_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
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
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "created.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "created.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "first"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    task = {"metadata": {"workspace_preflight": {"git": {"head": ""}}}}

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


def test_workspace_patch_excludes_sensitive_untracked_file_but_keeps_tracked_diff(tmp_path):
    """One credential-shaped untracked NAME must not annihilate the whole patch:
    the file is a disclosed per-file exclusion, the tracked diff survives (#447)."""
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    (repo / ".npmrc").write_text("//registry.npmjs.org/:_authToken=secret\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"

    artifacts, manifest = write_workspace_patch_artifacts(repo, artifact_dir, task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["errors"] == []
    assert manifest["sensitive_blocked"][0]["path"] == ".npmrc"
    assert any(item["kind"] == "workspace_patch" for item in artifacts)
    patch_text = (artifact_dir / "workspace.patch").read_text(encoding="utf-8")
    assert "tracked.txt" in patch_text
    assert "_authToken" not in patch_text


def test_workspace_patch_excludes_public_pem_and_disclosed_keeps_tracked_work(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    (repo / "public.pem").write_text("-----BEGIN PUBLIC KEY-----\nAAAA\n-----END PUBLIC KEY-----\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"

    artifacts, manifest = write_workspace_patch_artifacts(repo, artifact_dir, task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["sensitive_blocked"] == [
        {"path": "public.pem", "reason": "private key or certificate"}
    ]
    assert any(item["kind"] == "workspace_patch" for item in artifacts)
    patch_text = (artifact_dir / "workspace.patch").read_text(encoding="utf-8")
    assert "tracked.txt" in patch_text
    assert "public.pem" not in patch_text


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


def test_workspace_patch_excludes_sensitive_untracked_file_inside_excluded_dir(tmp_path):
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    secret = repo / "node_modules" / "pkg" / "service-account.json"
    secret.parent.mkdir(parents=True)
    secret.write_text("TOKEN=secret\n", encoding="utf-8")

    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})

    assert manifest["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert manifest["counts"]["sensitive_blocked"] == 1
    assert manifest["sensitive_blocked"][0]["path"] == "node_modules/pkg/service-account.json"
    assert any(item["kind"] == "workspace_patch" for item in artifacts)


def test_workspace_patch_excludes_common_credential_paths_but_still_produces_patch(tmp_path):
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
        "credentials",
        "prod.env",
        "settings.env.local",
        ".aws/credentials",
    }
    assert any(item["kind"] == "workspace_patch" for item in artifacts)
    patch_text = (artifact_dir / "workspace.patch").read_text(encoding="utf-8")
    assert "SECRET" not in patch_text


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
    result = json.loads((parent / "task_results" / "task-stale.json").read_text(encoding="utf-8"))
    assert any(item.get("kind") == "workspace_patch" for item in result["artifacts"])

    # A sensitive-shaped file no longer fails the patch (#447); break the HEAD
    # ref instead to force a genuinely FAILED refinalization.
    head_ref = subprocess.run(["git", "symbolic-ref", "--quiet", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / ".git" / head_ref).unlink()
    finalize_task_artifacts(parent, task)

    result = json.loads((parent / "task_results" / "task-stale.json").read_text(encoding="utf-8"))
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

    result = json.loads((parent / "task_results" / "task-head.json").read_text(encoding="utf-8"))
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

    result = json.loads((parent / "task_results" / "task-exception.json").read_text(encoding="utf-8"))
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
