"""File-only and mixed Git results preserve complete bytes and conflict semantics."""
from hashlib import sha256
import os
from pathlib import Path

import pytest

from ouroboros import workspace_patch_capture as capture
from ouroboros.workspace_file_outputs import (
    file_output_changes, prepare_file_outputs, verify_file_outputs,
)
from tests._workspace_git import commit, git, publish_capture, repository


@pytest.mark.parametrize("unborn", [False, True])
def test_binary_file_only_result_applies_and_rolls_back_without_touching_index(tmp_path, unborn):
    source = repository(tmp_path / "source", unborn=unborn)
    if unborn:
        target = repository(tmp_path / "target", unborn=True)
    else:
        target = tmp_path / "target"
        git(tmp_path, "clone", str(source), str(target))
    (source / "nested").mkdir()
    (source / "nested" / "answer.bin").write_bytes(b"answer\x00bytes")
    (target / "unrelated.txt").write_text("owner dirty content")
    git(target, "add", "unrelated.txt")
    index_before = git(target, "ls-files", "--stage", "-z")
    _, manifest = capture.write_workspace_patch_artifacts(source, tmp_path / "capture", task={})
    assert manifest["status"] == "ready_with_changes", manifest["errors"]
    assert manifest["patch_size"] == 0
    assert not (tmp_path / "capture" / "workspace.patch").exists()
    rows = file_output_changes(manifest, tmp_path / "capture")
    assert len(rows) == 1 and rows[0]["before"] is None
    with prepare_file_outputs(rows, target, baseline_sha=manifest["base_head"]) as prepared:
        assert prepared.paths == ["nested/answer.bin"]
        prepared.apply()
        assert prepared.verify_applied()
        assert (target / "nested" / "answer.bin").read_bytes() == b"answer\x00bytes"
        prepared.rollback()
    assert not (target / "nested" / "answer.bin").exists()
    assert (target / "unrelated.txt").read_text() == "owner dirty content"
    assert git(target, "ls-files", "--stage", "-z") == index_before


@pytest.mark.parametrize("operation", ["modify", "delete", "symlink"])
def test_tracked_large_changes_have_exact_preimages_and_complete_apply(tmp_path, monkeypatch, operation):
    source = repository(tmp_path / "source")
    original = b"large baseline\x00content"
    (source / "data.bin").write_bytes(original)
    (source / "data.bin").chmod(0o755)
    baseline = commit(source)
    target = tmp_path / "target"
    git(tmp_path, "clone", str(source), str(target))
    index_before = git(target, "ls-files", "--stage", "-z")
    if operation == "modify":
        (source / "data.bin").write_bytes(b"new binary\x00result")
        (source / "data.bin").chmod(0o644)
    elif operation == "delete":
        (source / "data.bin").unlink()
    elif operation == "rename":
        git(source, "mv", "data.bin", "renamed.bin")
    else:
        (source / "data.bin").unlink()
        try:
            (source / "data.bin").symlink_to("other-target.bin")
        except OSError as exc:
            pytest.skip(str(exc))
    monkeypatch.setattr(capture, "_PATCH_FILE_REFERENCE_BYTES", 8)
    _, manifest = capture.write_workspace_patch_artifacts(source, tmp_path / "capture", task={})
    assert manifest["status"] == "ready_with_changes", manifest["errors"]
    assert manifest["patch_size"] == 0
    rows = file_output_changes(manifest, tmp_path / "capture")
    prior = next(row["before"] for row in rows if row["path"] == "data.bin")
    assert prior["size"] == len(original) and prior["sha256"] == sha256(original).hexdigest()
    with prepare_file_outputs(rows, target, baseline_sha=baseline) as prepared:
        prepared.apply()
        assert verify_file_outputs(rows, target)
        if operation == "modify":
            assert (target / "data.bin").read_bytes() == b"new binary\x00result"
            if os.name != "nt":
                assert (target / "data.bin").stat().st_mode & 0o777 == 0o644
        elif operation == "rename":
            assert (target / "renamed.bin").read_bytes() == original
            assert not (target / "data.bin").exists()
        elif operation == "symlink":
            assert (target / "data.bin").is_symlink()
            assert os.readlink(target / "data.bin") == "other-target.bin"
        else:
            assert not (target / "data.bin").exists()
        prepared.rollback()
    assert (target / "data.bin").read_bytes() == original
    assert git(target, "ls-files", "--stage", "-z") == index_before
    assert not git(target, "status", "--porcelain")


def two_file_capture(tmp_path):
    source = repository(tmp_path / "source")
    target = repository(tmp_path / "target")
    for name in ("one.bin", "two.bin"):
        (source / name).write_bytes(name.encode() + b"\x00answer")
    cap_dir = tmp_path / "capture"
    _, manifest = capture.write_workspace_patch_artifacts(source, cap_dir, task={})
    return target, manifest, cap_dir


def test_all_target_paths_checked_before_any_file_apply(tmp_path):
    target, manifest, cap_dir = two_file_capture(tmp_path)
    rows = file_output_changes(manifest, cap_dir)
    with prepare_file_outputs(rows, target) as prepared:
        (target / "two.bin").write_bytes(b"concurrent edit")
        with pytest.raises(ValueError, match="changed before apply"):
            prepared.apply()
    assert not (target / "one.bin").exists()
    assert (target / "two.bin").read_bytes() == b"concurrent edit"


def test_each_target_checked_again_when_an_editor_changes_later_file_during_copy(tmp_path, monkeypatch):
    from ouroboros import artifacts
    target, manifest, cap_dir = two_file_capture(tmp_path)
    rows = file_output_changes(manifest, cap_dir)
    with prepare_file_outputs(rows, target) as prepared:
        original_copy = artifacts.copy_artifact_file
        def copy_with_other_editor(source, destination, **kwargs):
            if Path(destination) == target / "one.bin":
                (target / "two.bin").write_bytes(b"concurrent edit during first copy")
            return original_copy(source, destination, **kwargs)
        monkeypatch.setattr(artifacts, "copy_artifact_file", copy_with_other_editor)
        with pytest.raises(ValueError, match="changed during apply: two.bin"):
            prepared.apply()
        assert (target / "one.bin").read_bytes() == b"one.bin\x00answer"
        assert (target / "two.bin").read_bytes() == b"concurrent edit during first copy"
        prepared.rollback()
    assert not (target / "one.bin").exists()
    assert (target / "two.bin").read_bytes() == b"concurrent edit during first copy"


def test_malformed_side_returns_explicit_value_error(tmp_path):
    target, manifest, cap_dir = two_file_capture(tmp_path)
    manifest["file_output_changes"][0]["before"] = "unobserved"
    with pytest.raises(ValueError, match="no complete identity"):
        file_output_changes(manifest, cap_dir)


def test_corrupt_second_member_never_applies_first_file(tmp_path):
    import zipfile
    from ouroboros.artifacts import stream_artifact_file
    target, manifest, cap_dir = two_file_capture(tmp_path)
    archive = next(row for row in manifest["file_outputs"] if row["kind"] == "workspace_file_outputs")
    with zipfile.ZipFile(cap_dir / archive["name"], "w") as zipped:
        zipped.writestr("one.bin", b"one.bin\x00answer")
        zipped.writestr("two.bin", b"wrong bytes")
    archive.update(stream_artifact_file(cap_dir / archive["name"]))
    rows = file_output_changes(manifest, cap_dir)
    with pytest.raises(OSError, match="verification"):
        with prepare_file_outputs(rows, target):
            pytest.fail("corrupt source must not reach apply")
    assert not (target / "one.bin").exists()
    assert not (target / "two.bin").exists()


def test_rollback_preserves_concurrent_post_apply_edits(tmp_path):
    target, manifest, cap_dir = two_file_capture(tmp_path)
    with prepare_file_outputs(file_output_changes(manifest, cap_dir), target) as prepared:
        prepared.apply()
        (target / "one.bin").write_bytes(b"owner newer version")
        with pytest.raises(ValueError, match="preserved concurrent changes"):
            prepared.rollback()
    assert (target / "one.bin").read_bytes() == b"owner newer version"
    assert not (target / "two.bin").exists()


def test_target_parent_escape_and_incomplete_old_capture_fail_explicitly(tmp_path):
    target, manifest, cap_dir = two_file_capture(tmp_path)
    rows = file_output_changes(manifest, cap_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (target / "nested").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(str(exc))
    rows[0]["path"] = "nested/one.bin"
    with pytest.raises(ValueError, match="escapes target"):
        with prepare_file_outputs(rows, target):
            pytest.fail("escaped path")
    del manifest["file_output_changes"]
    with pytest.raises(ValueError, match="baseline unavailable"):
        file_output_changes(manifest, cap_dir)


def test_full_large_payload_is_streamed_through_capture_prepare_and_apply(tmp_path, monkeypatch):
    from ouroboros.artifacts import stream_artifact_file
    source = repository(tmp_path / "source")
    target = repository(tmp_path / "target")
    original_read = Path.read_bytes
    def no_large_read(path):
        assert path.stat().st_size <= 50 * 1024 * 1024, "large file read into one bytes object"
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", no_large_read)
    with (source / "large.bin").open("wb") as handle:
        for _ in range(52):
            handle.write(b"\x00payload" * (1024 * 128))
    expected = stream_artifact_file(source / "large.bin")
    cap_dir = tmp_path / "capture"
    _, manifest = capture.write_workspace_patch_artifacts(source, cap_dir, task={})
    assert manifest["status"] == "ready_with_changes", manifest["errors"]
    with prepare_file_outputs(file_output_changes(manifest, cap_dir), target) as prepared:
        prepared.apply()
        assert stream_artifact_file(target / "large.bin") == expected


def test_text_patch_and_file_result_deliver_together_with_explicit_git_staging(tmp_path):
    source = repository(tmp_path / "source")
    (source / "code.txt").write_text("old code\n")
    baseline = commit(source)
    target = tmp_path / "target"
    git(tmp_path, "clone", str(source), str(target))
    (source / "code.txt").write_text("new code\n")
    (source / "answer.bin").write_bytes(b"complete\x00answer")
    cap_dir = tmp_path / "capture"
    _, manifest = capture.write_workspace_patch_artifacts(source, cap_dir, task={})
    assert manifest["status"] == "ready_with_changes" and manifest["patch_size"] > 0
    with prepare_file_outputs(file_output_changes(manifest, cap_dir), target, baseline_sha=baseline) as prepared:
        git(target, "apply", "--index", str(cap_dir / "workspace.patch"))
        prepared.apply()
        git(target, "add", "--", *prepared.paths)
    assert (target / "code.txt").read_text() == "new code\n"
    assert git(target, "show", ":answer.bin") == b"complete\x00answer"
    assert not git(target, "diff", "--name-only")
    assert set(git(target, "diff", "--cached", "--name-only").decode().splitlines()) == {"answer.bin", "code.txt"}


def publish_child_patch(tmp_path, monkeypatch, shape="mixed", *, text_change=None, producer_layout=None):
    """Publish the child through ordinary host finalization, registering its exact files."""
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.registry import ToolContext

    drive = tmp_path / "data"
    drive.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(drive))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    target = repository(tmp_path / "target")
    if text_change is not None:
        (target / "answer.txt").write_bytes(text_change[0])
        commit(target)
    child = tmp_path / "child"
    git(tmp_path, "clone", "-q", str(target), str(child))
    task = {"parent_task_id": "parent", "workspace_root": str(child), "workspace_mode": "self_worktree",
            "task_constraint": {"mode": "acting_subagent", "surface": "self_worktree", "write_root": str(child),
                                "base_sha": git(child, "rev-parse", "HEAD").decode().strip()}}
    contents = {}
    if shape != "binary":
        contents["answer.txt"] = text_change[1] if text_change is not None else b"child text\n"
    if shape != "text":
        contents["archive.bin"] = b"\x00child archive"
    for name, data in contents.items():
        (child / name).write_bytes(data)
    producer = drive
    if producer_layout:
        from ouroboros.task_results import write_task_result

        producer = drive / producer_layout
        producer.mkdir(parents=True)
        task.update(task_attempt=1, started_at="2026-01-01T00:00:01Z")
        write_task_result(drive, "child", "running", **task, child_drive_root=str(producer), artifact_status="pending")
    publish_capture(producer, {"id": "child", **task})
    artifacts = load_task_result(producer, "child")["artifacts"]
    ctx = ToolContext(repo_dir=target, drive_root=drive, task_id="parent")
    return ctx, target, contents, {row["kind"]: row for row in artifacts}


@pytest.mark.parametrize("shape", ["text", "binary", "mixed"])
@pytest.mark.parametrize("decision", ["reject", "apply"])
def test_published_patch_decisions_count_text_and_file_results(tmp_path, monkeypatch, shape, decision):
    import json
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.tools.subagent_integration import _compare_subagent_patches, _integrate_subagent_patch

    ctx, target, contents, records = publish_child_patch(tmp_path, monkeypatch, shape)
    drive = ctx.drive_root
    assert ("workspace_patch" in records) == ("answer.txt" in contents)
    compared = _compare_subagent_patches(ctx, ["child"])
    assert "patch status: ready_with_changes" in compared
    assert ("1 separate file result(s) retained" in compared) == ("archive.bin" in contents)
    assert "EXCLUDED" not in compared and git(target, "status", "--porcelain") == b""
    if shape == "binary" and decision == "reject":
        from ouroboros import cli
        from tests._task_file_http import task_file_client, use_cli_http

        with task_file_client(drive) as http:
            reader, _ = use_cli_http(monkeypatch, http)
            row = reader.request("GET", "/api/tasks/child")
            with pytest.raises(cli.PatchCLIError, match="complete files.*workspace_patch.json"):
                cli._patch_from_result(reader, "child", row, strict=True)

    out = _integrate_subagent_patch(ctx, task_id="child", decision=decision, reason="selected after comparison")
    verdict = json.loads((task_artifact_dir_path(drive, "parent") / "subagent_patch_verdict_child.json").read_text())
    assert verdict["files"] == sorted(contents)
    assert "EXCLUDED" not in out
    if decision == "reject":
        assert f"{len(contents)} intended path(s), none applied or staged" in out
        assert (verdict["outcome"], verdict["applied"]) == ("rejected", False)
        assert git(target, "status", "--porcelain") == b""
    else:
        assert f"{len(contents)} file(s), staged" in out
        assert ("1 separate file result(s) delivered and staged" in out) == ("archive.bin" in contents)
        assert (verdict["outcome"], verdict["applied"]) == ("applied", True)
        for name, data in contents.items():
            assert git(target, "show", f":{name}") == (target / name).read_bytes() == data


@pytest.mark.parametrize("layout, stage", [
    ("state/headless_tasks/child/data", "before-copyback"), ("task_drives/child", "partial-copyback"),
    ("task_drives/child", "canonical-references"),
])
def test_split_capture_reads_the_published_files_before_copyback(tmp_path, monkeypatch, layout, stage):
    from ouroboros import cli
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.subagent_integration import _compare_subagent_patches, _integrate_subagent_patch
    from tests._task_file_http import task_file_client, use_cli_http

    ctx, target, contents, records = publish_child_patch(tmp_path, monkeypatch, producer_layout=layout)
    canonical = task_artifact_dir_path(ctx.drive_root, "child")
    producer = ctx.drive_root / layout
    requested = "child"
    if stage == "canonical-references":
        row = load_task_result(producer, "child")
        fields = {key: row[key] for key in ("artifacts", "artifact_status", "artifact_bundle", "artifact_finalized_at")}
        write_task_result(ctx.drive_root, "child", "completed", **fields)
        requested = "previous"
        write_task_result(ctx.drive_root, requested, "failed", parent_task_id="parent", superseded_by="child")
    if stage == "partial-copyback":
        previous = Path(records["workspace_patch"]["path"]).read_bytes()
        contents["answer.txt"] = b"later child text\n"
        row = load_task_result(producer, "child")
        (Path(row["workspace_root"]) / "answer.txt").write_bytes(contents["answer.txt"])
        write_task_result(producer, "child", "completed", artifact_status="finalizing")
        publish_capture(producer, {"id": "child"})
        records = {item["kind"]: item for item in load_task_result(producer, "child")["artifacts"]}
        assert records["workspace_patch_manifest"]["relpath"].startswith("captures/")
        assert Path(records["workspace_patch"]["path"]).read_bytes() != previous
        for kind in ("workspace_patch_manifest", "workspace_file_outputs"):
            destination = canonical / records[kind]["relpath"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            Path(records[kind]["path"]).replace(destination)
    expected = Path(records["workspace_patch"]["path"]).read_bytes()
    with task_file_client(ctx.drive_root) as http:
        reader, _ = use_cli_http(monkeypatch, http)
        public = reader.request("GET", f"/api/tasks/{requested}")
        assert public["status"] == "completed" and public["task_id"] == "child"
        assert public["workspace_capture"]["status"] == public["artifact_status"] == "ready_with_changes"
        assert cli._patch_bytes_from_result(reader, requested, public, strict=True) == expected
    assert "+" + contents["answer.txt"].decode().strip() in _compare_subagent_patches(ctx, [requested])
    out = _integrate_subagent_patch(ctx, task_id=requested)
    assert "✅ Integrated" in out, out
    for name, data in contents.items():
        assert git(target, "show", f":{name}") == data
    assert not (canonical / records["workspace_patch"].get("relpath", "workspace.patch")).exists()
    assert load_task_result(ctx.drive_root, "child")["artifact_status"] == (
        "ready_with_changes" if stage == "canonical-references" else "pending")


@pytest.mark.parametrize("case", ["attempt", "start", "other_layout", "reopened", "finalizing", "native_pending", "readonly"])
def test_split_capture_rejects_a_replica_outside_the_current_publication(tmp_path, monkeypatch, case):
    from ouroboros import cli
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.subagent_integration import _compare_subagent_patches, _integrate_subagent_patch
    from tests._task_file_http import task_file_client, use_cli_http

    layout = "retained-copy" if case == "other_layout" else "task_drives/child"
    ctx, target, _, records = publish_child_patch(tmp_path, monkeypatch, producer_layout=layout)
    changes = {
        "attempt": {"task_attempt": 2}, "start": {"started_at": "2026-01-01T01:00:00Z"},
        "other_layout": {}, "finalizing": {"artifact_status": "finalizing"},
        "native_pending": {"native_capture_pending": {"run": {"effect": "unknown"}}},
        "readonly": {"delegation_role": "subagent", "task_constraint": {"mode": "local_readonly_subagent"}},
        "reopened": {"artifacts": [{**records["workspace_patch_manifest"], "kind": "workspace_capture_history"}]},
    }
    write_task_result(ctx.drive_root, "child", "running", **changes[case])
    with task_file_client(ctx.drive_root) as http:
        reader, _ = use_cli_http(monkeypatch, http)
        public = reader.request("GET", "/api/tasks/child")
        assert public["workspace_capture"]["status"] in {"pending", "finalizing"}
        with pytest.raises(cli.PatchCLIError):
            cli._patch_bytes_from_result(reader, "child", public, strict=True)
    assert "+child text" not in _compare_subagent_patches(ctx, ["child"])
    assert "✅" not in _integrate_subagent_patch(ctx, task_id="child")
    assert git(target, "status", "--porcelain") == b""


def test_legacy_file_results_under_ignore_rules_land_beside_the_text_patch(tmp_path, monkeypatch):
    """#2446 on the copy-less legacy integration: the text patch applies with --3way --index and a file result
    the target already tracks under an ignored folder stages; a new file its ignore rules exclude, from the
    text patch or a file result alike, is delivered, kept out of the index and named in the same receipt."""
    import json
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    target = repository(tmp_path / "target")
    (target / ".gitignore").write_bytes(b".build/\ndist/\n")
    (target / "code.txt").write_bytes(b"old\n")
    (target / ".build").mkdir()
    (target / ".build" / "pinned.bin").write_bytes(b"pinned\x00baseline")
    git(target, "add", "-f", "--", ".build/pinned.bin")
    base = commit(target)
    child = tmp_path / "child"
    git(tmp_path, "clone", "-q", str(target), str(child))
    (child / "code.txt").write_bytes(b"new\n")
    (child / ".build" / "pinned.bin").write_bytes(b"pinned\x00by the child")
    (child / "dist").mkdir()
    (child / "dist" / "new.bin").write_bytes(b"new\x00file result")
    (child / "dist" / "notes.txt").write_bytes(b"n\n")  # small text: it rides the text patch
    git(child, "add", "-f", "--", "dist/new.bin", "dist/notes.txt")
    monkeypatch.setattr(capture, "_PATCH_FILE_REFERENCE_BYTES", 8)
    artifacts = task_artifact_dir_path(data, "child", create=True)
    _, manifest = capture.write_workspace_patch_artifacts(
        child, artifacts, task={"metadata": {"workspace_preflight": {"git": {"head": base}}}})
    assert manifest["status"] == "ready_with_changes" and manifest["patch_size"] > 0, manifest
    assert {row["path"] for row in file_output_changes(manifest, artifacts)} == {".build/pinned.bin", "dist/new.bin"}
    write_task_result(data, "child", "completed", parent_task_id="parent",
                      task_constraint={"mode": "acting_subagent", "surface": "self_worktree"})
    out = _integrate_subagent_patch(ToolContext(repo_dir=target, drive_root=data, task_id="parent"), task_id="child")
    assert "✅ Integrated" in out and "4 file(s), 2 staged)" in out, out
    assert "Delivered, not indexed" in out and "dist/new.bin, dist/notes.txt" in out, out
    assert "dist/new.bin (captured as file-reference artifact) [unstaged]" in out and "EXCLUDED" not in out
    assert "Changes are staged" not in out and "the files named above are not staged at all" in out, out
    assert git(target, "show", ":code.txt") == b"new\n"
    assert git(target, "show", ":.build/pinned.bin") == b"pinned\x00by the child"
    assert (target / "dist" / "new.bin").read_bytes() == b"new\x00file result"
    assert (target / "dist" / "notes.txt").read_text() == "n\n"
    assert git(target, "ls-files", "--", "dist") == b""
    verdict = json.loads((task_artifact_dir_path(data, "parent") / "subagent_patch_verdict_child.json").read_text())
    assert (verdict["outcome"], verdict["unindexed"]) == ("applied", ["dist/new.bin", "dist/notes.txt"])


@pytest.mark.parametrize("failing", ["new_file_add", "flag_remark"])
def test_a_failed_legacy_staging_undoes_text_patch_file_results_and_index(tmp_path, monkeypatch, failing):
    """A real index failure while the copy-less legacy integration stages (another Git process holds
    index.lock), first at a new file's add, then at the last step that marks a staged entry assume-unchanged
    again: the text patch, the file results and every touched index entry and flag go back, the parent's own
    unrelated staged and unstaged work stays as it was, and the same captured result then lands."""
    import json
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import patch_staging as staging
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    target = repository(tmp_path / "target")
    (target / ".gitignore").write_bytes(b"dist/\n")
    for name, body in (("code.txt", b"old\n"), ("flag.txt", b"flag\n"), ("other.txt", b"base\n")):  # LF on every OS
        (target / name).write_bytes(body)
    (target / "pinned.bin").write_bytes(b"pinned\x00baseline")
    base = commit(target)
    child = tmp_path / "child"
    git(tmp_path, "clone", "-q", str(target), str(child))
    (target / "other.txt").write_bytes(b"base\nparent staged\n")  # the parent's own work, untouched by the child
    git(target, "add", "--", "other.txt")
    (target / "other.txt").write_bytes(b"base\nparent staged\nparent unstaged\n")
    git(target, "update-index", "--assume-unchanged", "--", "flag.txt")
    (child / "code.txt").write_bytes(b"new\n")
    (child / "flag.txt").write_bytes(b"flag2\n")
    (child / "fresh.txt").write_bytes(b"f\n")
    (child / "pinned.bin").write_bytes(b"pinned\x00by the child")
    (child / "dist").mkdir()
    (child / "dist" / "notes.txt").write_bytes(b"n\n")
    (child / "dist" / "new.bin").write_bytes(b"new\x00file result")
    git(child, "add", "-f", "--", "dist/new.bin", "dist/notes.txt")
    monkeypatch.setattr(capture, "_PATCH_FILE_REFERENCE_BYTES", 8)
    artifacts = task_artifact_dir_path(data, "child", create=True)
    _, manifest = capture.write_workspace_patch_artifacts(
        child, artifacts, task={"metadata": {"workspace_preflight": {"git": {"head": base}}}})
    assert manifest["status"] == "ready_with_changes" and manifest["patch_size"] > 0, manifest
    assert {row["path"] for row in file_output_changes(manifest, artifacts)} == {"pinned.bin", "dist/new.bin"}
    write_task_result(data, "child", "completed", parent_task_id="parent",
                      task_constraint={"mode": "acting_subagent", "surface": "self_worktree"})

    def state():
        files = {name: (target / name).read_bytes() if (target / name).exists() else None
                 for name in ("code.txt", "flag.txt", "other.txt", "pinned.bin", "fresh.txt",
                              "dist/notes.txt", "dist/new.bin")}
        return (git(target, "ls-files", "--stage", "-v", "-z"), git(target, "diff", "--cached", "--binary"),
                git(target, "diff-files", "--name-only", "-z"), files)

    before, real, calls = state(), staging._git_paths, []

    def locked(root, args, paths, prefix=b""):
        calls.append(args)
        step = (args[1:2] == ["add"] and not {"-u", "-N", "--refresh"} & set(args) if failing == "new_file_add"
                else "--assume-unchanged" in args and sum("--assume-unchanged" in seen for seen in calls) == 1)
        if step:
            lock = Path(root) / ".git" / "index.lock"
            lock.write_bytes(b"")
            try:
                return real(root, args, paths, prefix)
            finally:
                lock.unlink()
        return real(root, args, paths, prefix)

    ctx = ToolContext(repo_dir=target, drive_root=data, task_id="parent")
    monkeypatch.setattr(staging, "_git_paths", locked)
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "INTEGRATE_CONFLICT" in out and "index.lock" in out and "nothing is half-applied" in out, out
    assert "separate file result(s) rolled back" in out and "EXCLUDED" not in out
    assert state() == before
    verdict = json.loads((task_artifact_dir_path(data, "parent") / "subagent_patch_verdict_child.json").read_text())
    assert (verdict["outcome"], verdict["applied"]) == ("conflict", False)
    monkeypatch.setattr(staging, "_git_paths", real)  # the lock is gone: the same captured result now lands
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "✅ Integrated" in out and "6 file(s), 4 staged)" in out, out
    assert git(target, "show", ":code.txt") == b"new\n" and git(target, "show", ":fresh.txt") == b"f\n"
    assert git(target, "show", ":flag.txt") == b"flag2\n" and git(target, "ls-files", "-v", "--", "flag.txt")[:1] == b"h"
    assert git(target, "show", ":pinned.bin") == b"pinned\x00by the child"
    assert git(target, "ls-files", "--", "dist") == b"" and (target / "dist" / "new.bin").exists()
    assert git(target, "show", ":other.txt") == b"base\nparent staged\n"
    assert (target / "other.txt").read_text() == "base\nparent staged\nparent unstaged\n"


def use_finished_cli_task(monkeypatch, reader):
    """Use the published child for submission. Detail and artifact requests still go through the gateway."""
    from ouroboros import cli

    request = reader.request

    def submitted(method, path, *args, **kwargs):
        if (method, path) == ("POST", "/api/tasks"):
            return {"task_id": "child"}
        return request(method, path, *args, **kwargs)

    monkeypatch.setattr(reader, "request", submitted)
    monkeypatch.setattr(cli, "_client", lambda *a, **kw: reader)
    monkeypatch.setattr(cli, "_await_cost_finality", lambda client, tid, result: result)


def test_cli_patch_out_preserves_non_utf8_patch_bytes(tmp_path, monkeypatch, capsys):
    from ouroboros import cli
    from tests._task_file_http import task_file_client, use_cli_http

    ctx, _, _, records = publish_child_patch(tmp_path, monkeypatch, "text",
                                            text_change=(b"menu\n", b"caf\xe9\n"))
    record = records["workspace_patch"]
    captured = Path(record["path"]).read_bytes()
    assert b"+caf\xe9\n" in captured and sha256(captured).hexdigest() == record["sha256"]
    output = tmp_path / "export.patch"
    with task_file_client(ctx.drive_root) as http:
        reader, requests = use_cli_http(monkeypatch, http)
        use_finished_cli_task(monkeypatch, reader)
        assert cli.main(["run", "--no-stream", "--patch", "--patch-out", str(output), "fixture"]) == 0
    assert output.read_bytes() == captured
    assert capsys.readouterr().out == captured.decode("utf-8", errors="replace")
    assert [path for method, path in requests if "/artifacts/" in path] == [
        "/api/tasks/child/artifacts/workspace_patch.json", "/api/tasks/child/artifacts/workspace.patch"]


@pytest.mark.parametrize("status, damage", [
    ("ready", ""), ("ready", "clean"), ("ready_with_changes", "other_missing"),
    ("ready_with_changes", "measured_other_missing"), ("ready_with_changes", "patch_missing"),
    ("ready_with_changes", "manifest_size"), ("ready_with_changes", "measured_manifest"),
    ("ready_with_changes", "duplicate_manifest"),
])
def test_finalized_fixed_name_capture_keeps_its_published_identities(tmp_path, monkeypatch, status, damage):
    import json
    from ouroboros import cli
    from ouroboros.outcomes import artifact_bundle_from_result
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch
    from tests._task_file_http import task_file_client, use_cli_http

    ctx, target, contents, records = publish_child_patch(tmp_path, monkeypatch, "text")
    path = Path(records["workspace_patch_manifest"]["path"])
    manifest = json.loads(path.read_bytes())
    manifest["status"] = status
    fields = ("kind", "name", "path", "workspace_root")
    patch = {k: records["workspace_patch"][k] for k in (*fields, "size", "sha256")}
    if damage == "clean":
        Path(patch["path"]).write_bytes(b"")
        patch.update(size=0, sha256=sha256(b"").hexdigest())
        manifest.update(patch_size=0, sha256=patch["sha256"], untracked_included=[], diffstat="")
        manifest["counts"]["untracked_included"] = 0
    path.write_text(json.dumps(manifest), encoding="utf-8")
    # The old writer published the manifest's byte count and the patch's SHA-256.
    header = {k: records["workspace_patch_manifest"][k] for k in fields}
    header["size"] = path.stat().st_size
    rows = [header, patch]
    if damage == "duplicate_manifest":
        rows.append(dict(header))
    if damage == "measured_other_missing":
        header["sha256"] = sha256(path.read_bytes()).hexdigest()
    if damage.endswith("other_missing"):
        rows.append({"kind": "task_artifact", "name": "missing.txt", "path": str(path.parent / "missing.txt")})
    bundle = artifact_bundle_from_result({"artifacts": rows, "artifact_status": status})
    if damage == "measured_manifest":
        bundle["artifacts"][0]["sha256"] = sha256(path.read_bytes()).hexdigest()
    write_task_result(ctx.drive_root, "child", "completed", artifacts=rows, artifact_status=status, artifact_bundle=bundle)
    captured = Path(patch["path"]).read_bytes()
    if damage == "manifest_size":
        path.write_bytes(path.read_bytes() + b" ")
    if damage == "patch_missing":
        Path(patch["path"]).unlink()
    with task_file_client(ctx.drive_root) as http:
        reader, _ = use_cli_http(monkeypatch, http)
        public = reader.request("GET", "/api/tasks/child")
        if damage.endswith("missing"):
            assert public["artifact_status"] == "missing"
        if damage == "clean":
            assert cli._patch_bytes_from_result(reader, "child", public, strict=False) == b""
            assert "INTEGRATE_NO_CHANGES" in _integrate_subagent_patch(ctx, task_id="child")
        elif not damage or damage.endswith("other_missing"):
            assert cli._patch_bytes_from_result(reader, "child", public, strict=True) == captured
            assert "✅ Integrated" in _integrate_subagent_patch(ctx, task_id="child")
            assert (target / "answer.txt").read_bytes() == contents["answer.txt"]
        else:
            with pytest.raises(cli.PatchCLIError):
                cli._patch_bytes_from_result(reader, "child", public, strict=True)
            assert "✅" not in _integrate_subagent_patch(ctx, task_id="child")
            assert git(target, "status", "--porcelain") == b""


def _encode_fixed_git_capture(drive, records, *, historical):
    """Serialize a real text capture using the fixed writer's published fields (41a320cb9bc)."""
    import json
    from ouroboros.artifacts import stream_artifact_file
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import atomic_write_json

    path = Path(records["workspace_patch_manifest"]["path"])
    manifest = json.loads(path.read_bytes())
    assert manifest["status"] in {"ready_no_changes", "ready_with_changes"} and not manifest["errors"]
    assert not manifest["file_outputs"] and not manifest["file_output_changes"]
    if historical:
        keys = ("schema_version", "created_at", "workspace_root", "patch_name", "manifest_name",
                "patch_size", "sha256", "diffstat", "counts", "tracked_changed", "untracked_included",
                "untracked_excluded", "sensitive_blocked", "exclude_rules_version", "diagnostics", "errors")
        manifest = {key: manifest[key] for key in keys}
        manifest["counts"].pop("tracked_excluded")
        manifest["exclude_rules_version"] = 1
    manifest["status"] = "ready"
    patch = path.with_name("workspace.patch")
    if not manifest["patch_size"]:
        patch.write_bytes(b"")  # the fixed writer opened this file even on a clean tree
    measured = stream_artifact_file(patch)
    assert measured["size"] == manifest["patch_size"]
    manifest["sha256"] = measured["sha256"]
    atomic_write_json(path, manifest, trailing_newline=True)
    root = {"workspace_root": manifest["workspace_root"]}
    rows = [
        {**root, "name": patch.name, "path": str(patch), "kind": "workspace_patch", **measured},
        {**root, "name": path.name, "path": str(path), "kind": "workspace_patch_manifest", "size": path.stat().st_size},
    ]
    if not historical:
        rows[1].update(stream_artifact_file(path))
    write_task_result(drive, "child", "completed", artifacts=rows, artifact_status="ready", artifact_bundle=None)


@pytest.mark.parametrize("form, clean, damage", [
    ("current", True, ""), ("current", False, ""), ("fixed", True, ""), ("fixed", False, ""),
    ("malformed", True, ""), ("malformed", False, ""),
    ("fixed", False, "manifest_hash"), ("fixed", False, "patch_size"), ("fixed", True, "conflict"),
])
def test_git_capture_after_drive_retirement(tmp_path, monkeypatch, form, clean, damage):
    from ouroboros import cli
    from ouroboros.artifacts import artifact_record, task_artifact_dir_path
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch, _locate_child_patch
    from tests._store_shared import add_artifact_conflict
    from tests._task_file_http import task_file_client, use_cli_http
    from tests._workspace_git import retire_published_capture

    before, after = b"base\n", b"child\n"
    ctx, target, _, records = publish_child_patch(
        tmp_path, monkeypatch, "text", text_change=(before, before if clean else after),
        producer_layout="state/headless_tasks/child/data")
    child = ctx.drive_root / "state" / "headless_tasks" / "child" / "data"
    expected = b"" if clean else Path(records["workspace_patch"]["path"]).read_bytes()
    if form != "current":
        _encode_fixed_git_capture(child, records, historical=form == "fixed")
    retire_published_capture(ctx.drive_root, child, "child")
    store = task_artifact_dir_path(ctx.drive_root, "child")
    patch, header = store / "workspace.patch", store / "workspace_patch.json"
    if damage == "manifest_hash":
        raw = header.read_bytes()
        assert raw.endswith(b"\n")
        header.write_bytes(raw[:-1] + b" ")  # same JSON value and length
    elif damage == "patch_size":
        patch.write_bytes(expected + b" ")
    elif damage == "conflict":
        add_artifact_conflict(ctx.drive_root, "child", artifact_record(patch, kind="workspace_patch"))
    _, view, _ = _locate_child_patch(ctx, "child")
    with task_file_client(ctx.drive_root) as http:
        reader, _ = use_cli_http(monkeypatch, http)
        result = reader.request("GET", "/api/tasks/child")
        if form == "malformed" or damage:
            assert view["status"] == "failed"
            with pytest.raises(cli.PatchCLIError):
                cli._patch_bytes_from_result(reader, "child", result, strict=False)
            message = _integrate_subagent_patch(ctx, task_id="child")
            assert "✅" not in message and "INTEGRATE_NO_CHANGES" not in message
            assert (target / "answer.txt").read_bytes() == before
            return
        assert view["status"] == ("ready_no_changes" if clean else "ready_with_changes")
        assert cli._patch_bytes_from_result(reader, "child", result, strict=not clean) == expected
        if clean:
            with pytest.raises(cli.PatchCLIError, match="no changes"):
                cli._patch_bytes_from_result(reader, "child", result, strict=True)
    message = _integrate_subagent_patch(ctx, task_id="child")
    assert ("INTEGRATE_NO_CHANGES" if clean else "✅ Integrated") in message, message
    assert (target / "answer.txt").read_bytes() == (before if clean else after)


@pytest.mark.parametrize("mismatch", [{"task_id": "another"}, {"task_attempt": 2}])
def test_capture_replica_must_match_task_and_attempt(tmp_path, monkeypatch, mismatch):
    from ouroboros.task_results import load_task_result
    from ouroboros.workspace_capture_result import read_replica

    ctx, _, _, _ = publish_child_patch(tmp_path, monkeypatch, "text", producer_layout="task_drives/child")
    canonical = load_task_result(ctx.drive_root, "child")
    child = ctx.drive_root / "task_drives" / "child"
    _, admitted = read_replica(ctx.drive_root, "child", canonical, child, load_task_result)
    assert admitted == child

    def mismatched_row(drive, task_id):
        return {**load_task_result(drive, task_id), **mismatch}

    _, admitted = read_replica(ctx.drive_root, "child", canonical, child, mismatched_row)
    assert admitted is None


@pytest.mark.parametrize("case", ["published", "size_changed", "unpublished", "pending"])
def test_cli_reads_patch_only_from_the_pre_manifest_finalizer(tmp_path, monkeypatch, case):
    from ouroboros import cli
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_results import write_task_result
    from tests._task_file_http import task_file_client, use_cli_http

    path = task_artifact_dir_path(tmp_path, "child", create=True) / "workspace.patch"
    raw = b"diff --git a/a b/a\n"
    path.write_bytes(raw)
    root = str(tmp_path / "work")
    rows = [{"kind": "workspace_patch", "path": str(path), "size": len(raw), "workspace_root": root}]
    lifecycle = {"artifact_status": "pending"} if case == "pending" else {}
    write_task_result(tmp_path, "child", "completed", workspace_root=root,
                      artifacts=[] if case == "unpublished" else rows, **lifecycle)
    if case == "size_changed":
        path.write_bytes(raw + b"more")
    with task_file_client(tmp_path) as http:
        reader, _ = use_cli_http(monkeypatch, http)
        public = reader.request("GET", "/api/tasks/child")
        if case == "published":
            assert cli._patch_bytes_from_result(reader, "child", public, strict=True) == raw
        else:
            with pytest.raises(cli.PatchCLIError):
                cli._patch_bytes_from_result(reader, "child", public, strict=True)


@pytest.mark.parametrize("conflict", [
    "other", "other_with_patch_basename", "other_with_patch_display_name", "workspace_patch_manifest", "workspace_patch",
    "workspace_file_outputs_manifest", "workspace_file_outputs",
])
def test_only_the_current_captures_own_conflict_blocks_its_consumers(tmp_path, monkeypatch, capsys, conflict):
    from ouroboros import cli
    from ouroboros.artifacts import artifact_record, task_artifact_dir_path
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.subagent_integration import _compare_subagent_patches, _integrate_subagent_patch
    from tests._store_shared import add_artifact_conflict
    from tests._task_file_http import task_file_client, use_cli_http

    ctx, target, contents, records = publish_child_patch(tmp_path, monkeypatch)
    drive = ctx.drive_root
    unrelated = conflict.startswith("other")
    if unrelated:
        path = task_artifact_dir_path(drive, "child") / {
            "other": "recipe.md", "other_with_patch_basename": "notes/workspace.patch",
            "other_with_patch_display_name": "saved-workspace.patch",
        }[conflict]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("check the captured result\n", encoding="utf-8")
        record = artifact_record(path)
        if conflict == "other_with_patch_display_name":
            # A retained ordinary copy can keep its original name when its store address changes.
            record["name"] = "workspace.patch"
    else:
        record = records[conflict]
    identities = add_artifact_conflict(drive, "child", record)
    comparison = _compare_subagent_patches(ctx, ["child"])
    with task_file_client(drive) as http:
        reader, requests = use_cli_http(monkeypatch, http)
        public = reader.request("GET", "/api/tasks/child")
        assert public["artifact_bundle"]["status"] == public["artifact_status"] == "failed"
        assert not cli._is_terminal_success(public)
        assert public["workspace_capture"]["status"] == "ready_with_changes"
        use_finished_cli_task(monkeypatch, reader)
        output = tmp_path / "selected.patch"
        assert cli.main(["run", "--no-stream", "--patch-out", str(output), "fixture"]) == 1
        diagnostic = capsys.readouterr().err
        if unrelated:
            assert output.read_bytes() == Path(records["workspace_patch"]["path"]).read_bytes()
            for kind in ("workspace_file_outputs_manifest", "workspace_file_outputs"):
                assert ("HEAD", f"/api/tasks/child/artifacts/{records[kind]['name']}") in requests
        else:
            assert not output.exists()

    out = _integrate_subagent_patch(ctx, task_id="child")
    for shown in (comparison, diagnostic, out):
        assert Path(record["path"]).name in shown and all(digest in shown for digest in identities), shown
    if unrelated:
        assert "+child text" in comparison and "✅ Integrated" in out
        for name, data in contents.items():
            assert git(target, "show", f":{name}") == (target / name).read_bytes() == data
    else:
        assert "+child text" not in comparison and "patch status: failed" in comparison
        assert "INTEGRATE_CAPTURE_FAILED" in out and "INTEGRATE_NO_CHANGES" not in out
        assert git(target, "status", "--porcelain") == b""
        assert not (task_artifact_dir_path(drive, "parent") / "subagent_patch_verdict_child.json").exists()
        if conflict == "workspace_file_outputs":
            rejected = _integrate_subagent_patch(ctx, task_id="child", decision="reject", reason="unavailable file result")
            assert "2 intended path(s), none applied or staged" in rejected
            assert git(target, "status", "--porcelain") == b""
    kept = {row.get("sha256") for row in load_task_result(drive, "child")["artifacts"]
            if row.get("path") == record["path"]}
    assert set(identities) <= kept


@pytest.mark.parametrize("damage", [
    "manifest_bytes", "patch_bytes", "registered_patch_identity", "manifest_missing", "patch_missing",
    "file_outputs_missing", "file_outputs_manifest_missing", "producer_failed", "producer_ready_no_changes",
    "producer_finalizing", "producer_missing", "producer_blank", "history_only",
    "unregistered_workspace_patch_manifest", "unregistered_workspace_patch", "unregistered_workspace_file_outputs",
])
def test_an_invalid_current_capture_never_falls_back_to_saved_bytes(tmp_path, monkeypatch, damage):
    from ouroboros import cli
    from ouroboros.artifacts import artifact_record, task_artifact_dir_path
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.subagent_integration import _compare_subagent_patches, _integrate_subagent_patch
    from tests._store_shared import add_artifact_conflict
    from tests._task_file_http import task_file_client, use_cli_http

    ctx, target, _, records = publish_child_patch(tmp_path, monkeypatch)
    drive = ctx.drive_root
    path = task_artifact_dir_path(drive, "child") / "recipe.md"
    path.write_text("independent artifact\n", encoding="utf-8")
    add_artifact_conflict(drive, "child", artifact_record(path))
    row = load_task_result(drive, "child")
    if damage.startswith("producer_"):
        status = "" if damage == "producer_blank" else damage.removeprefix("producer_")
        write_task_result(drive, "child", "completed", artifact_status=status)
    elif damage.startswith("unregistered_"):
        missing_kind = damage.removeprefix("unregistered_")
        write_task_result(drive, "child", "completed", artifacts=[r for r in row["artifacts"] if r["kind"] != missing_kind])
    elif damage == "history_only":
        history = [{**r, "capture_kind": r["kind"], "kind": "workspace_capture_history"}
                   if r["kind"] in records else r for r in row["artifacts"]]
        write_task_result(drive, "child", "completed", artifacts=history)
    elif damage == "registered_patch_identity":
        updated = [{**r, "sha256": "a" * 64} if r["kind"] == "workspace_patch" else r for r in row["artifacts"]]
        write_task_result(drive, "child", "completed", artifacts=updated)
    else:
        kind = {"manifest": "workspace_patch_manifest", "patch": "workspace_patch",
                "file_outputs": "workspace_file_outputs", "file_outputs_manifest": "workspace_file_outputs_manifest"}[
                    damage.rsplit("_", 1)[0]]
        artifact = Path(records[kind]["path"])
        if damage.endswith("_missing"):
            artifact.unlink()
        else:
            # Whitespace leaves JSON parseable; its published bytes are still different.
            artifact.write_bytes(artifact.read_bytes() + b" ")

    compared = _compare_subagent_patches(ctx, ["child"])
    assert "+child text" not in compared and "patch status: ready_with_changes" not in compared
    with task_file_client(drive) as http:
        reader, _ = use_cli_http(monkeypatch, http)
        public = reader.request("GET", "/api/tasks/child")
        with pytest.raises(cli.CLIError):
            cli._patch_from_result(reader, "child", public, strict=True)
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "✅" not in out and "INTEGRATE_NO_CHANGES" not in out and "INTEGRATE_" in out, out
    assert git(target, "status", "--porcelain") == b""
    assert not (task_artifact_dir_path(drive, "parent") / "subagent_patch_verdict_child.json").exists()


def test_cli_checks_file_result_bytes_on_head_without_spooling_them(tmp_path, monkeypatch):
    from ouroboros import artifacts, cli
    from tests._task_file_http import task_file_client, use_cli_http

    ctx, _, _, records = publish_child_patch(tmp_path, monkeypatch)
    archive = records["workspace_file_outputs"]
    verify = artifacts.stream_artifact_file
    sinks = []

    def observe(source, sink=None, *, expected=None):
        if hasattr(source, "fileno") and expected and expected.get("sha256") == archive["sha256"]:
            sinks.append(sink)
        return verify(source, sink, expected=expected)

    monkeypatch.setattr(artifacts, "stream_artifact_file", observe)
    with task_file_client(ctx.drive_root) as http:
        reader, requests = use_cli_http(monkeypatch, http)
        public = reader.request("GET", "/api/tasks/child")
        assert cli._patch_bytes_from_result(reader, "child", public, strict=True)
        assert ("HEAD", f"/api/tasks/child/artifacts/{archive['name']}") in requests
        assert sinks == [None]
        path = Path(archive["path"])
        path.write_bytes(b"x" * path.stat().st_size)
        with pytest.raises(cli.PatchCLIError):
            cli._patch_bytes_from_result(reader, "child", public, strict=True)
        assert sinks == [None, None]


@pytest.mark.parametrize("header, wrong", [
    ("x-ouroboros-artifact-sha256", "a" * 64), ("content-length", "0"),
])
def test_cli_requires_both_verified_headers_before_returning_the_text_patch(tmp_path, monkeypatch, header, wrong):
    from ouroboros import cli
    from tests._task_file_http import task_file_client, use_cli_http

    ctx, _, _, records = publish_child_patch(tmp_path, monkeypatch)
    archive = records["workspace_file_outputs"]
    with task_file_client(ctx.drive_root) as http:
        reader, requests = use_cli_http(monkeypatch, http)
        public = reader.request("GET", "/api/tasks/child")
        request = http.request

        def with_mismatched_header(method, url, **kwargs):
            response = request(method, url, **kwargs)
            if method == "HEAD" and response.request.url.path.endswith("/" + archive["name"]):
                assert response.status_code == 200
                assert response.headers["x-ouroboros-artifact-sha256"] == archive["sha256"]
                assert response.headers["content-length"] == str(archive["size"])
                response.headers[header] = wrong
            return response

        monkeypatch.setattr(http, "request", with_mismatched_header)
        with pytest.raises(cli.PatchCLIError):
            cli._patch_from_result(reader, "child", public, strict=True)
        assert ("HEAD", f"/api/tasks/child/artifacts/{archive['name']}") in requests
