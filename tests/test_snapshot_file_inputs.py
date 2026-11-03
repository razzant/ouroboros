"""Snapshot inputs are git's own: binary inputs ride the snapshot commit, the copy
holds git's checkout representation, and a result is a normal binary patch.

Every input of a private copy, a binary document included, is a blob of the
snapshot commit B in the source's object database (unreachable once the copy's
pins go, so ``git gc`` removes it). The copy is git's checkout of B, so its bytes
are the source repository's checkout representation (attributes, eol and the
mode git tracks), an untouched copy captures as an empty result, and a changed
input comes back as a ``git apply --binary`` patch that refuses a concurrent edit.
"""
from pathlib import Path
import os
import subprocess
import zipfile

import pytest

from ouroboros.subagent_worktrees import capture_result_commit, find_execution_snapshot, provision_execution_snapshot


def git(root, *args, check=True):
    return subprocess.run(["git", *args], cwd=root, check=check, capture_output=True)


def target_tree(tmp_path, geometry="ordinary"):
    root = tmp_path / "target"
    root.mkdir()
    git(root, "init")
    (root / "code.txt").write_text("committed\n", encoding="utf-8")
    git(root, "add", "code.txt")
    if geometry != "unborn":
        git(root, "-c", "user.name=Fixture", "-c", "user.email=f@invalid", "commit", "-m", "base")
    if geometry == "linked":
        linked = tmp_path / "linked"
        git(root, "worktree", "add", "--detach", str(linked))
        root = linked
    (root / "code.txt").write_text("staged\n", encoding="utf-8")
    git(root, "add", "code.txt")
    (root / "code.txt").write_text("unstaged\n", encoding="utf-8")
    return root


def document(path, text="Original document"):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        archive.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        archive.writestr("word/document.xml", f'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>')


def snapshot(tmp_path, target):
    return provision_execution_snapshot(
        target_root=target, task_id="input-test", snapshot_id="input-snapshot",
        worktree_root=tmp_path / "snapshots", data_dir=tmp_path / "data",
    )


def capture(tmp_path, handle):
    """The run's result C and its patch; ``(result, patch path, changed paths)``."""
    patch = tmp_path / "capture" / "workspace.patch"
    result = capture_result_commit(handle.snapshot_id, patch, data_dir=tmp_path / "data")
    changed = git(handle.path, "diff-tree", "-r", "--no-renames", "--name-only", handle.baseline_sha,
                  result["result_sha"]).stdout.decode().split()
    return result, patch, changed


@pytest.mark.parametrize("geometry", ["ordinary", "unborn", "linked"])
@pytest.mark.parametrize("operation", ["modify", "delete", "unchanged"])
def test_a_document_input_rides_the_snapshot_and_its_binary_result_applies_back(tmp_path, geometry, operation):
    target = target_tree(tmp_path, geometry)
    document(target / "input.docx")
    original = (target / "input.docx").read_bytes()
    blob = git(target, "hash-object", "input.docx").stdout.decode().strip()
    before_status = git(target, "status", "--porcelain=v1", "-z").stdout
    before_index = git(target, "ls-files", "--stage", "-z").stdout
    before_head = git(target, "rev-parse", "--verify", "HEAD", check=False).stdout
    handle = snapshot(tmp_path, target)
    execution = Path(handle.path)
    assert (execution / "input.docx").read_bytes() == original
    # The document is one blob of B in the source's object database; the source itself is unchanged.
    assert git(execution, "rev-parse", "HEAD:input.docx").stdout.decode().strip() == blob
    assert git(target, "cat-file", "-e", blob, check=False).returncode == 0
    assert git(target, "status", "--porcelain=v1", "-z").stdout == before_status
    assert git(target, "ls-files", "--stage", "-z").stdout == before_index
    assert git(target, "rev-parse", "--verify", "HEAD", check=False).stdout == before_head
    if operation == "modify":
        document(execution / "input.docx", "Changed document")
    elif operation == "delete":
        (execution / "input.docx").unlink()
    result, patch, changed = capture(tmp_path, handle)
    if operation == "unchanged":
        assert result["result_tree"] == handle.baseline_tree and patch.read_bytes() == b"" and changed == []
    else:
        assert changed == ["input.docx"] and b"GIT binary patch" in patch.read_bytes()
        git(target, "apply", "--binary", "--check", str(patch))
        git(target, "apply", "--binary", str(patch))
        if operation == "modify":
            assert (target / "input.docx").read_bytes() == (execution / "input.docx").read_bytes()
        else:
            assert not (target / "input.docx").exists()
        git(target, "apply", "--binary", "-R", str(patch))  # the same patch takes it back exactly
    assert (target / "input.docx").read_bytes() == original
    assert git(target, "ls-files", "--stage", "-z").stdout == before_index
    assert git(target, "status", "--porcelain=v1", "-z").stdout == before_status
    assert find_execution_snapshot(handle.snapshot_id, tmp_path / "data") is not None
    assert execution.exists()


def test_a_concurrent_owner_edit_of_a_document_input_refuses_the_binary_result(tmp_path):
    target = target_tree(tmp_path)
    document(target / "input.docx")
    handle = snapshot(tmp_path, target)
    document(Path(handle.path) / "input.docx", "Prepared child result")
    _result, patch, changed = capture(tmp_path, handle)
    assert changed == ["input.docx"]
    git(target, "apply", "--binary", "--check", str(patch))  # over the snapshot's bytes it applies
    document(target / "input.docx", "Concurrent owner edit")
    newer = (target / "input.docx").read_bytes()
    assert git(target, "apply", "--binary", "--check", str(patch), check=False).returncode != 0
    assert (target / "input.docx").read_bytes() == newer


def test_child_staging_unchanged_input_does_not_create_a_result(tmp_path):
    target = target_tree(tmp_path)
    document(target / "input.docx")
    handle = snapshot(tmp_path, target)
    git(handle.path, "add", "input.docx")
    result, patch, changed = capture(tmp_path, handle)
    assert result["result_tree"] == handle.baseline_tree and patch.read_bytes() == b"" and changed == []
    document(Path(handle.path) / "input.docx", "Changed and staged")
    git(handle.path, "add", "input.docx")
    _result, _patch, changed = capture(tmp_path, handle)
    assert changed == ["input.docx"], "a staged change is a result"


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("policy", ["autocrlf", "attributes", "info_attributes", "info_overrides_tree", "attributes_file"])
def test_git_checkout_keeps_git_s_representation_and_normal_patch_semantics(tmp_path, newline, policy):
    target = target_tree(tmp_path)
    git(target, "config", "core.autocrlf", "true" if policy == "autocrlf" else "false")
    if policy == "attributes":
        (target / ".gitattributes").write_bytes(b"*.txt text eol=crlf\n")
    elif policy == "info_attributes":  # repository-local: a linked worktree shares it
        (target / ".git" / "info").mkdir(exist_ok=True)
        (target / ".git" / "info" / "attributes").write_bytes(b"*.txt text\n")
        git(target, "config", "core.eol", "crlf")
    elif policy == "info_overrides_tree":  # info/attributes outranks the tree's own file
        (target / ".gitattributes").write_bytes(b"*.txt -text\n")
        (target / ".git" / "info").mkdir(exist_ok=True)
        (target / ".git" / "info" / "attributes").write_bytes(b"*.txt text eol=crlf\n")
    elif policy == "attributes_file":
        (tmp_path / "attributes").write_bytes(b"*.txt text eol=crlf\n")
        git(target, "config", "core.attributesFile", str(tmp_path / "attributes"))
    source_config = (target / ".git" / "config").read_bytes()
    original = newline.join([b"original", b"working bytes", b""])
    for name in ("code.txt", "staged.txt", "untracked.txt"):
        (target / name).write_bytes(original)
    git(target, "add", "staged.txt")
    (target / "deleted.txt").write_bytes(b"removed before snapshot\n")
    git(target, "add", "deleted.txt")
    (target / "deleted.txt").unlink()
    before_status = git(target, "status", "--porcelain=v1", "-z").stdout
    before_index = git(target, "ls-files", "--stage", "-z").stdout
    before_head = git(target, "rev-parse", "HEAD").stdout

    handle = snapshot(tmp_path, target)
    execution = Path(handle.path)
    for name in ("code.txt", "staged.txt", "untracked.txt"):
        # The copy is git's checkout: the source repository's checkout representation of B.
        representation = git(target, "cat-file", "--filters", f"{handle.baseline_sha}:{name}").stdout
        assert (execution / name).read_bytes() == representation == b"original\r\nworking bytes\r\n", name
        assert git(execution, "cat-file", "--filters", f"{handle.baseline_sha}:{name}").stdout == representation
    assert not (execution / "deleted.txt").exists()
    assert git(execution, "status", "--porcelain=v1", "-z").stdout == b""
    assert (target / ".git" / "config").read_bytes() == source_config
    assert git(target, "status", "--porcelain=v1", "-z").stdout == before_status
    assert git(target, "ls-files", "--stage", "-z").stdout == before_index
    assert git(target, "rev-parse", "HEAD").stdout == before_head
    assert all((target / name).read_bytes() == original
               for name in ("code.txt", "staged.txt", "untracked.txt"))
    result, patch, changed = capture(tmp_path, handle)
    assert result["result_tree"] == handle.baseline_tree and patch.read_bytes() == b"" and changed == []

    changed_bytes = original.replace(b"original", b"child edit")
    (execution / "code.txt").write_bytes(changed_bytes)
    _result, patch, changed = capture(tmp_path, handle)
    assert changed == ["code.txt"]
    git(target, "apply", "--binary", "--check", str(patch))
    git(target, "apply", "--binary", str(patch))
    # Explicit apply retains the target's normal Git checkout representation.
    assert (target / "code.txt").read_bytes() == b"child edit\r\nworking bytes\r\n"
    assert (target / "staged.txt").read_bytes() == original
    assert (target / "untracked.txt").read_bytes() == original
    assert git(target, "ls-files", "--stage", "-z").stdout == before_index
    assert git(target, "rev-parse", "HEAD").stdout == before_head


@pytest.mark.skipif(os.name == "nt", reason="no executable bit for the source to distrust")
def test_untrusted_executable_bit_stays_out_of_the_snapshot_and_its_diff(tmp_path):
    target = target_tree(tmp_path)
    git(target, "config", "core.filemode", "false")
    (target / "code.txt").chmod(0o755)  # not a change the source's Git can see
    handle = snapshot(tmp_path, target)
    execution = Path(handle.path)
    assert git(execution, "ls-tree", "HEAD", "code.txt").stdout.startswith(b"100644 ")
    assert not (execution / "code.txt").stat().st_mode & 0o111, "the copy has the mode git tracks"
    assert git(execution, "status", "--porcelain=v1", "-z").stdout == b""
    result, patch, changed = capture(tmp_path, handle)
    assert result["result_tree"] == handle.baseline_tree and changed == []
    (execution / "code.txt").write_bytes(b"child edit\n")
    (execution / "code.txt").chmod(0o755)
    _result, patch, changed = capture(tmp_path, handle)
    assert changed == ["code.txt"] and b"child edit" in patch.read_bytes()
    assert b"mode" not in patch.read_bytes()


def test_snapshot_keeps_git_link_entries_git_owned(tmp_path):
    target = target_tree(tmp_path)
    # Git's portable symlink checkout is a regular file containing the target;
    # the baseline mode, not its host file type, still owns that representation.
    git(target, "config", "core.symlinks", "false")
    (target / "link").write_bytes(b"code.txt")
    oid = git(target, "hash-object", "-w", "link").stdout.decode().strip()
    git(target, "update-index", "--add", "--cacheinfo", f"120000,{oid},link")
    commit = git(target, "rev-parse", "HEAD").stdout.decode().strip()
    git(target, "update-index", "--add", "--cacheinfo", f"160000,{commit},nested")
    git(target, "-c", "user.name=Fixture", "-c", "user.email=f@invalid", "commit", "-m", "link entries")
    (target / "nested").mkdir()  # An uninitialized gitlink stays Git-owned.
    before_index = git(target, "ls-files", "--stage", "-z").stdout

    handle = snapshot(tmp_path, target)
    execution = Path(handle.path)
    assert (execution / "link").read_bytes() == b"code.txt"
    assert git(execution, "ls-tree", "HEAD", "link").stdout.startswith(b"120000 blob ")
    assert git(execution, "ls-tree", "HEAD", "nested").stdout.startswith(b"160000 commit ")
    assert git(target, "ls-files", "--stage", "-z").stdout == before_index
    result, patch, changed = capture(tmp_path, handle)
    assert result["result_tree"] == handle.baseline_tree and patch.read_bytes() == b"" and changed == []
