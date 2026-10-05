"""Large ordinary file custody must stay streaming, complete and verifiable."""
from ouroboros.headless import retry_child_task_refs
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from ouroboros import artifacts


def _large_file(path):
    block = b"ordinary dataset\x00" * 65536
    digest = sha256()
    with path.open("wb") as handle:
        for _ in range(52):
            handle.write(block)
            digest.update(block)
    return {"size": len(block) * 52, "sha256": digest.hexdigest()}


def test_large_copy_version_and_directory_are_streamed(tmp_path, monkeypatch):
    source = tmp_path / "dataset"
    source.mkdir()
    large = source / "large.bin"
    expected = _large_file(large)
    assert expected["size"] > 50 * 1024 * 1024
    (source / "notes.txt").write_text("dataset explanation")
    ctx = SimpleNamespace(drive_root=tmp_path / "data", task_id="large")
    original = Path.read_bytes

    def no_large_read(path):
        if path.stat().st_size > 50 * 1024 * 1024:
            pytest.fail("large artifact was read into RAM as one bytes object")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", no_large_read)
    copied = artifacts.copy_file_to_task_artifacts(ctx, large)
    assert {key: copied[key] for key in expected} == expected
    with large.open("ab") as handle:
        handle.write(b"new version")
    artifacts.copy_file_to_task_artifacts(ctx, large)
    versions = list((ctx.drive_root / "task_results" / "artifact_versions" / ctx.task_id / copied["name"]).iterdir())
    assert len(versions) == 1
    assert artifacts.stream_artifact_file(versions[0]) == expected
    records = artifacts.copy_directory_to_task_artifacts(ctx, source)
    manifest = json.loads(Path(records[0]["path"]).read_text())
    assert manifest["file_count"] == 2
    with zipfile.ZipFile(records[1]["path"]) as archive:
        assert set(archive.namelist()) == {"large.bin", "notes.txt"}
        for row in manifest["files"]:
            digest, size = sha256(), 0
            with archive.open(row["path"]) as member:
                for chunk in iter(lambda: member.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            assert (size, digest.hexdigest()) == (row["size"], row["sha256"])


def test_changed_source_never_overwrites_previous_destination(tmp_path, monkeypatch):
    source, target = tmp_path / "source.bin", tmp_path / "durable.bin"
    source.write_bytes(b"old source")
    target.write_bytes(b"kept destination")
    expected = artifacts.stream_artifact_file(source)
    source.write_bytes(b"new source")
    with pytest.raises(OSError, match="verification"):
        artifacts.copy_artifact_file(source, target, expected=expected)
    assert target.read_bytes() == b"kept destination"
    assert not list(tmp_path.glob(".*.tmp"))


def test_source_change_during_stream_is_explicit(tmp_path):
    source = tmp_path / "mutable.bin"
    source.write_bytes(b"first")

    class ChangingSink:
        def write(self, chunk):
            source.write_bytes(b"second")

    with pytest.raises(OSError, match="changed"):
        artifacts.stream_artifact_file(source, ChangingSink())


@pytest.mark.parametrize("replace_link", [False, True])
def test_source_replacement_keeps_previous_destination(tmp_path, monkeypatch, replace_link):
    source, target = tmp_path / "source.bin", tmp_path / "durable.bin"
    source.write_bytes(b"original")
    target.write_bytes(b"previous durable")
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"different")
    if replace_link:
        link = tmp_path / "source-link.bin"
        try:
            link.symlink_to(source)
        except OSError:
            pytest.skip("symlinks unavailable")
        source = link
    original_stream = artifacts.stream_artifact_file
    replacement_refusals = []

    def replaced_during_copy(path, sink=None, **kwargs):
        class ReplacingSink:
            def write(self, chunk):
                if replace_link:
                    source.unlink()
                    source.symlink_to(replacement)
                else:
                    try:
                        replacement.replace(source)
                    except PermissionError as exc:
                        replacement_refusals.append(exc)
                        raise
                return sink.write(chunk)
        return original_stream(path, ReplacingSink(), **kwargs)

    monkeypatch.setattr(artifacts, "stream_artifact_file", replaced_during_copy)
    with pytest.raises(OSError) as caught:
        artifacts.copy_artifact_file(source, target)
    if replacement_refusals:
        # Windows may refuse replacing an open source. Prove that exact OS
        # refusal propagated and neither source file was replaced or consumed.
        assert caught.value is replacement_refusals[0]
        assert source.read_bytes() == b"original"
        assert replacement.read_bytes() == b"different"
    else:
        assert "changed" in str(caught.value)
        assert source.read_bytes() == b"different"  # replacement really happened
    assert target.read_bytes() == b"previous durable"
    assert not list(tmp_path.glob(".*.tmp"))


def test_borrowed_completed_spool_copies_prefix_and_retains_caller_ownership(tmp_path):
    import tempfile

    payload = b"prefix\x00" + b"content" * 10000
    with tempfile.SpooledTemporaryFile(max_size=100) as spool:
        spool.write(payload)
        spool.seek(7)
        target = tmp_path / "spooled.bin"
        measured = artifacts.copy_artifact_file(spool, target)
        assert measured == {"size": len(payload), "sha256": sha256(payload).hexdigest()}
        assert target.read_bytes() == payload
        assert not spool.closed
    assert spool.closed


@pytest.mark.parametrize("gap_kind", ["walk", "changing"])
def test_automatic_genesis_listing_gaps_do_not_fail_completed_capture(tmp_path, monkeypatch, gap_kind):
    import errno
    from ouroboros import headless
    from ouroboros.task_results import load_task_result, write_task_result

    root, workspace = tmp_path / "data", tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("delivered answer")
    changing = workspace / "growing.log"
    changing.write_bytes(b"before")
    task = {"id": "genesis", "workspace_root": str(workspace),
            "task_constraint": {"surface": "genesis"}}
    write_task_result(root, task["id"], "completed", result="delivered answer")
    # Isolate the automatic listing from the independently tested strict patch
    # capture. A listing gap must not rewrite an already successful capture axis.
    monkeypatch.setattr(headless, "write_workspace_patch_artifacts",
                        lambda *args, **kwargs: ([], {"status": "ready_no_changes"}))
    if gap_kind == "walk":
        original_walk = headless.os.walk
        def walk(path, *, onerror):
            onerror(PermissionError(errno.EACCES, "controlled unreadable directory", str(workspace / "private")))
            yield from original_walk(path, onerror=onerror)
        monkeypatch.setattr(headless.os, "walk", walk)
    else:
        original_stream = artifacts.stream_artifact_file
        writes = []
        def stream(path, sink=None, **kwargs):
            if Path(path) == changing:
                class GrowingSink:
                    def write(self, chunk):
                        # A finite writer keeps a regression bounded; the reader
                        # must reject the initial-size breach before this ends.
                        if len(writes) < 64:
                            with changing.open("ab") as handle:
                                handle.write(b"more")
                            writes.append(len(chunk))
                return original_stream(path, GrowingSink(), **kwargs)
            return original_stream(path, sink, **kwargs)
        monkeypatch.setattr(artifacts, "stream_artifact_file", stream)
    returned = headless.finalize_task_artifacts(root, task)
    entry = next(item for item in returned if item["kind"] == "deliverable_manifest")
    manifest = json.loads(Path(entry["path"]).read_text())
    assert manifest["complete"] is False and manifest["gap_count"] == 1
    assert manifest["truncated"] is False
    if gap_kind == "changing":
        assert len(writes) <= 2, "do not wait for an actively growing file to reach EOF"
    gaps = [item for item in manifest["contents"] if item.get("status") == "unavailable"]
    assert len(gaps) == 1 and "sha256" not in gaps[0]
    assert next(item for item in manifest["contents"] if item["rel"] == "answer.txt")["sha256"] == sha256(b"delivered answer").hexdigest()
    result = load_task_result(root, task["id"])
    assert result["status"] == "completed"
    assert result["artifact_status"] == result["artifact_bundle"]["status"] == "ready_no_changes"
    bundle_entry = next(item for item in result["artifact_bundle"]["artifacts"] if item["kind"] == "deliverable_manifest")
    assert bundle_entry["errors"] == ["Automatic workspace listing is partial: 1 read gaps."]


def test_missing_directory_member_is_not_silently_omitted(tmp_path):
    directory = tmp_path / "output"
    directory.mkdir()
    good, missing = directory / "good.txt", directory / "gone.txt"
    good.write_text("kept")
    ctx = SimpleNamespace(drive_root=tmp_path / "data", task_id="directory")
    with pytest.raises(OSError, match="unavailable"):
        artifacts.copy_directory_to_task_artifacts(ctx, directory, member_paths=[good, missing])
    root = artifacts.task_artifact_dir_path(ctx.drive_root, ctx.task_id)
    assert not list(root.iterdir())


def test_copyback_failure_retains_child_until_existing_retry_finishes(tmp_path, monkeypatch):
    from ouroboros.headless import copy_child_task_result, prepare_task_drive, remove_subagent_task_drive
    from ouroboros.observability import retry_pending_child_ref_promotions
    from ouroboros.task_results import write_task_result, load_task_result

    parent = tmp_path / "canonical"
    child = prepare_task_drive(parent, "custody", "empty")
    source = child / "generated.bin"
    source.write_bytes(b"complete generated artifact")
    record = artifacts.copy_file_to_task_artifacts(SimpleNamespace(drive_root=child, task_id="custody"), source)
    write_task_result(child, "custody", "completed", result="done", artifacts=[record], artifact_status="ready")
    original = artifacts.copy_artifact_file

    def fail_canonical(src, dst, **kwargs):
        if Path(dst).is_relative_to(parent / "task_results"):
            raise OSError("injected destination failure")
        return original(src, dst, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "copy_artifact_file", fail_canonical)
        copied = copy_child_task_result(parent, {"id": "custody", "drive_root": str(child)})
        copied = retry_child_task_refs(parent, child, "custody")
    assert copied["artifact_bundle"]["status"] == "missing"
    assert copied["child_ref_promotion"]["status"] == "incomplete"
    assert not remove_subagent_task_drive(parent, "custody", live=lambda _task: False)
    assert Path(record["path"]).is_file()
    assert retry_pending_child_ref_promotions(parent)["completed"] == ["custody"]
    result = load_task_result(parent, "custody")
    assert result["child_ref_promotion"]["status"] == "complete"
    assert result["artifact_bundle"]["status"] == "ready"
    assert remove_subagent_task_drive(parent, "custody", live=lambda _task: False)
    assert Path(result["artifacts"][0]["path"]).read_bytes() == b"complete generated artifact"


def _many_inputs(tmp_path, drive, task_id, count=28):
    source = tmp_path / (task_id + "-inputs")
    source.mkdir()
    paths = []
    for index in range(count):
        path = source / f"input-{index:02}.txt"
        path.write_text(f"complete input {index}")
        paths.append({"path": str(path)})
    rows = artifacts.stage_task_attachments(drive, task_id, paths)
    assert len(rows) == count and all(row["status"] == "staged" for row in rows)
    return artifacts.attachment_manifest_projection(drive, task_id, rows)


def test_full_input_contract_inheritance_retry_and_mailbox(tmp_path):
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.owner_mailbox import write_owner_message, copy_owner_mailbox_for_retry, owner_attachment_manifest
    from ouroboros.tools.control_scheduling import _materialize_child_attachment_manifest, _build_child_subagent_contract
    import shutil

    drive = tmp_path / "data"
    authority = _many_inputs(tmp_path, drive, "parent")
    contract = build_task_contract({"id": "parent", "task_contract": authority})
    assert contract["attachment_manifest_ref"] == authority["attachment_manifest_ref"]
    assert len(contract["attachment_manifest"]) == 25
    mailbox = _many_inputs(tmp_path, drive, "mailbox")
    mailbox_rows = artifacts.resolve_attachment_manifest(drive, "mailbox", mailbox)
    captured, error = artifacts.materialize_inherited_attachment_manifest(mailbox_rows, drive, "parent")
    assert not error
    assert write_owner_message(drive, "use every attached input", "parent", attachment_manifest=captured)
    child_authority, error = _materialize_child_attachment_manifest(contract, drive, "child", owner_drive=drive, owner_task_id="parent")
    assert not error
    assert len(artifacts.resolve_attachment_manifest(drive, "child", child_authority)) == 56
    child_contract = _build_child_subagent_contract({"tid": "child", "parent_contract": contract, **child_authority})
    assert child_contract["attachment_manifest_ref"] == child_authority["attachment_manifest_ref"]
    assert child_contract["attachment_manifest_ref"] != contract["attachment_manifest_ref"]
    task = {"id": "retry", "task_contract": contract, "metadata": {"task_contract": contract}}
    replacements, error = artifacts.handoff_task_attachments_for_retry(drive, "parent", "retry", task)
    assert not error
    assert copy_owner_mailbox_for_retry(drive, "parent", "retry", path_replacements=replacements)
    shutil.rmtree(artifacts.task_artifact_dir_path(drive, "parent"))
    assert len(artifacts.resolve_attachment_manifest(drive, "retry", task["task_contract"])) == 28
    assert len(owner_attachment_manifest(drive, "retry")) == 28
    for row in artifacts.resolve_attachment_manifest(drive, "retry", task["task_contract"]):
        assert artifacts.stream_artifact_file(Path(row["abs_path"]), expected=row)["sha256"] == row["sha256"]


def test_input_manifest_tamper_and_changed_file_never_fall_back_to_preview(tmp_path):
    from ouroboros.tools.control_scheduling import _materialize_child_attachment_manifest

    drive = tmp_path / "data"
    authority = _many_inputs(tmp_path, drive, "original")
    rows = artifacts.resolve_attachment_manifest(drive, "original", authority)
    Path(rows[-1]["abs_path"]).write_text("changed input")
    copied, error = _materialize_child_attachment_manifest(authority, drive, "child", owner_drive=drive, owner_task_id="original")
    assert not copied and "verification" in error
    ref = authority["attachment_manifest_ref"]
    path = artifacts.task_artifact_dir_path(drive, "original") / ref["path"]
    path.write_bytes(b"[]")
    with pytest.raises(ValueError, match="verification"):
        artifacts.resolve_attachment_manifest(drive, "original", authority)


def test_full_inputs_survive_copyback_and_child_gc(tmp_path):
    from ouroboros.headless import copy_child_task_result, prepare_task_drive, remove_subagent_task_drive
    from ouroboros.task_results import write_task_result

    parent = tmp_path / "canonical"
    child = prepare_task_drive(parent, "inputs", "empty")
    authority = _many_inputs(tmp_path, child, "inputs")
    write_task_result(child, "inputs", "completed", result="done", task_contract=authority)
    result = copy_child_task_result(parent, {"id": "inputs", "drive_root": str(child)})
    result = retry_child_task_refs(parent, child, "inputs")
    assert result["child_ref_promotion"]["status"] == "complete"
    assert remove_subagent_task_drive(parent, "inputs", live=lambda _task: False)
    rows = artifacts.resolve_attachment_manifest(parent, "inputs", result["task_contract"])
    assert len(rows) == 28
    for row in rows:
        assert artifacts.stream_artifact_file(Path(row["abs_path"]), expected=row)


def test_large_workspace_output_is_a_file_reference_not_a_git_patch(tmp_path):
    import subprocess
    from ouroboros.workspace_patch_capture import write_workspace_patch_artifacts

    repo = tmp_path / "workspace"
    repo.mkdir()
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    git("init")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--allow-empty", "-m", "base")
    expected = _large_file(repo / "dataset.bin")
    output, manifest = write_workspace_patch_artifacts(repo, tmp_path / "capture", task={})
    assert not manifest["errors"]
    assert manifest["patch_size"] == 0
    assert manifest["status"] == "ready_with_changes"
    assert not (tmp_path / "capture" / "workspace.patch").exists()
    assert not any(row["kind"] == "workspace_patch" for row in output)
    assert len(manifest["file_outputs"]) == 2
    captured = next(row for row in output if row["kind"] == "workspace_file_outputs_manifest")
    contents = json.loads(Path(captured["path"]).read_text())
    assert contents["files"] == [{"path": "dataset.bin", **expected}]


def test_document_bridge_rejects_foreign_task_and_changed_capture(tmp_path, monkeypatch):
    from ouroboros.tools.core_artifacts import _send_file
    from supervisor import message_bus

    source = tmp_path / "report.txt"
    source.write_text("original delivered bytes")
    ctx = SimpleNamespace(drive_root=tmp_path / "data", task_id="document", task_metadata={},
                          current_chat_id=1, pending_events=[])
    assert _send_file(ctx, str(source)).startswith("OK")
    first = ctx.pending_events[0]
    monkeypatch.setattr(message_bus, "DATA_DIR", ctx.drive_root)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    published = []
    monkeypatch.setattr(message_bus, "publish_event", lambda kind, event: published.append(event))
    bridge = message_bus.LocalChatBridge({})
    assert bridge.send_document(1, b"", task_id=ctx.task_id, file_ref=first["file_ref"])[0]
    assert published[-1]["file_ref"] == first["file_ref"]
    assert not bridge.send_document(1, b"", task_id="other", file_ref=first["file_ref"])[0]
    source.write_text("later delivered bytes")
    assert _send_file(ctx, str(source)).startswith("OK")
    second = ctx.pending_events[-1]
    assert second["file_ref"]["path"] != first["file_ref"]["path"]
    from ouroboros.gateway.files import resolve_task_file_reference
    original = resolve_task_file_reference(ctx.drive_root, ctx.task_id, first["file_ref"])
    assert original.read_text() == "original delivered bytes"
    original.write_text("tampered")
    assert not bridge.send_document(1, b"", task_id=ctx.task_id, file_ref=first["file_ref"])[0]


@pytest.mark.parametrize("ref", [{}, "", False, {"kind": "task_source"}])
def test_malformed_full_reference_cannot_become_a_complete_preview(tmp_path, ref):
    authority = {"attachment_manifest": [{"status": "staged", "label": "only a preview"}],
                 "attachment_manifest_ref": ref}
    with pytest.raises((ValueError, TypeError)):
        artifacts.resolve_attachment_manifest(tmp_path, "invalid", authority)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_capture_does_not_reuse_mutable_external_file_alias(tmp_path, link_kind):
    import os
    source = tmp_path / "input.txt"
    source.write_text("captured bytes")
    drive = tmp_path / "data"
    store = artifacts.task_artifact_dir_path(drive, "alias", create=True)
    attachments = store / "attachments"
    attachments.mkdir()
    identity = artifacts.stream_artifact_file(source)
    immutable = store / f"input-{identity['sha256']}.txt"
    try:
        for target in (attachments / source.name, immutable):
            if link_kind == "symlink":
                target.symlink_to(source)
            else:
                os.link(source, target)
    except OSError as exc:
        pytest.skip(f"link creation unavailable: {exc}")
    rows = artifacts.stage_task_attachments(drive, "alias", [{"path": str(source)}])
    assert rows[0]["status"] == "staged"
    captured_input = Path(rows[0]["abs_path"])
    assert not captured_input.is_symlink() and not captured_input.samefile(source)
    output = artifacts.copy_file_to_task_artifacts(SimpleNamespace(drive_root=drive, task_id="alias"), source, immutable=True)
    captured_output = Path(output["path"])
    assert not captured_output.is_symlink() and not captured_output.samefile(source)
    source.write_text("later external mutation")
    assert captured_input.read_text() == captured_output.read_text() == "captured bytes"


def test_native_image_beyond_inline_preview_remains_visible(tmp_path):
    import base64
    from ouroboros.context import _build_attachment_image_blocks

    drive = tmp_path / "data"
    authority = _many_inputs(tmp_path, drive, "images", count=28)
    original = artifacts.resolve_attachment_manifest(drive, "images", authority)
    image = tmp_path / "last.png"
    image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="))
    image_rows = artifacts.stage_task_attachments(drive, "images", [{"path": str(image)}])
    authority = artifacts.attachment_manifest_projection(drive, "images", [*original, *image_rows])
    blocks = _build_attachment_image_blocks({"id": "images", "drive_root": str(drive), "task_contract": authority})
    assert any(block["type"] == "image_url" for block in blocks)
    assert blocks[-1]["_source_path"].endswith("last.png")


def test_staging_reads_fresh_bytes_when_source_preserves_its_mtime(tmp_path):
    import os
    source = tmp_path / "updated.txt"
    source.write_text("first")
    drive = tmp_path / "data"
    first = artifacts.stage_task_attachments(drive, "fresh", [{"path": str(source)}])[0]
    artifacts.stage_task_attachments(drive, "fresh", [{"path": str(source)}])
    observed = source.stat()
    source.write_text("later")
    os.utime(source, ns=(observed.st_atime_ns, observed.st_mtime_ns))
    latest = artifacts.stage_task_attachments(drive, "fresh", [{"path": str(source)}])[0]
    assert latest["abs_path"] != first["abs_path"]
    assert Path(first["abs_path"]).read_text() == "first"
    assert Path(latest["abs_path"]).read_text() == "later"


def test_large_tracked_symlink_replacement_is_not_misreported_as_deletion(tmp_path, monkeypatch):
    import subprocess
    from ouroboros import workspace_patch_capture as capture

    repo = tmp_path / "workspace"
    repo.mkdir()
    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    git("init")
    (repo / "dataset.bin").write_bytes(b"old large content")
    (repo / "target.txt").write_text("target")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "base")
    (repo / "dataset.bin").unlink()
    try:
        (repo / "dataset.bin").symlink_to("target.txt")
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    monkeypatch.setattr(capture, "_PATCH_FILE_REFERENCE_BYTES", 8)
    _, manifest = capture.write_workspace_patch_artifacts(repo, tmp_path / "capture", task={})
    row = next(row for row in manifest["tracked_excluded"] if row["path"] == "dataset.bin")
    assert row["symlink"] and row["link_target"] == "target.txt" and not row["deleted"]
    assert not manifest["errors"]


def test_large_staged_rename_keeps_both_paths_out_of_the_git_patch(tmp_path, monkeypatch):
    import subprocess
    from ouroboros import workspace_patch_capture as capture

    repo = tmp_path / "workspace"
    repo.mkdir()
    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    git("init")
    (repo / "dataset.bin").write_bytes(b"large captured contents")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "base")
    git("mv", "dataset.bin", "renamed.bin")
    monkeypatch.setattr(capture, "_PATCH_FILE_REFERENCE_BYTES", 8)
    _, manifest = capture.write_workspace_patch_artifacts(repo, tmp_path / "capture", task={})
    assert not manifest["errors"] and manifest["patch_size"] == 0
    assert {row["path"] for row in manifest["tracked_excluded"]} == {"dataset.bin", "renamed.bin"}
    assert len(manifest["file_outputs"]) == 2


def test_complete_directory_manifest_has_no_old_member_count_cutoff(tmp_path):
    from ouroboros.tools.shell_outputs import _directory_fingerprint
    directory = tmp_path / "many-members"
    directory.mkdir()
    for index in range(1002):
        (directory / f"member-{index:04}.txt").write_text(str(index))
    before = _directory_fingerprint(directory)
    (directory / "member-1001.txt").write_text("changed beyond the former bound")
    assert _directory_fingerprint(directory) != before
    ctx = SimpleNamespace(drive_root=tmp_path / "data", task_id="many")
    records = artifacts.copy_directory_to_task_artifacts(ctx, directory)
    manifest = json.loads(Path(records[0]["path"]).read_text())
    assert manifest["file_count"] == 1002
    assert {row["path"] for row in manifest["files"]} == {p.name for p in directory.iterdir()}
    with zipfile.ZipFile(records[1]["path"]) as archive:
        assert len(archive.namelist()) == 1002
        assert archive.read("member-1001.txt") == b"changed beyond the former bound"


def test_retry_rejects_missing_input_outside_inline_preview(tmp_path):
    drive = tmp_path / "data"
    authority = _many_inputs(tmp_path, drive, "missing")
    rows = artifacts.resolve_attachment_manifest(drive, "missing", authority)
    Path(rows[-1]["abs_path"]).unlink()
    task = {"id": "retry", "task_contract": authority}
    _, error = artifacts.handoff_task_attachments_for_retry(drive, "missing", "retry", task)
    assert error and task["task_contract"] == authority


@pytest.mark.parametrize("drive_kind", ["headless", "direct"])
def test_input_copy_failure_protects_each_existing_gc_root(tmp_path, monkeypatch, drive_kind):
    from ouroboros import headless
    from ouroboros.observability import retry_pending_child_ref_promotions
    from ouroboros.task_results import write_task_result, load_task_result
    parent = tmp_path / "canonical"
    child = (headless.prepare_task_drive(parent, "inputs", "empty") if drive_kind == "headless"
             else parent / headless.TASK_DRIVES_DIR / "inputs")
    child.mkdir(parents=True, exist_ok=True)
    authority = _many_inputs(tmp_path, child, "inputs")
    write_task_result(child, "inputs", "completed", result="answer", task_contract=authority,
                      completed_at="2000-01-01T00:00:00Z")
    original = artifacts.copy_artifact_file
    def unavailable(src, dst, **kwargs):
        if Path(dst).is_relative_to(parent / "task_results"):
            raise OSError("canonical storage interrupted")
        return original(src, dst, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "copy_artifact_file", unavailable)
        result = headless.copy_child_task_result(parent, {"id": "inputs", "drive_root": str(child)})
        result = retry_child_task_refs(parent, child, "inputs")
        assert result["child_ref_promotion"]["status"] == "incomplete"
        prune = headless.prune_headless_task_drives if drive_kind == "headless" else headless.prune_task_drives
        assert not prune(parent, retention_days=1, now=4_000_000_000)["pruned"]
        assert not headless.remove_subagent_task_drive(parent, "inputs", live=lambda _task: False)
        assert child.is_dir()
    assert retry_pending_child_ref_promotions(parent)["completed"] == ["inputs"]
    result = load_task_result(parent, "inputs")
    assert result["child_ref_promotion"]["status"] == "complete"
    assert headless.remove_subagent_task_drive(parent, "inputs", live=lambda _task: False)
    assert len(artifacts.resolve_attachment_manifest(parent, "inputs", result["task_contract"])) == 28


@pytest.mark.parametrize("immutable", [False, True])
def test_collection_preserves_capture_identity_without_freezing_mutable_outputs(tmp_path, immutable):
    """Writer-side collection keeps an immutable capture and discloses changed bytes,
    a mutable output takes its new identity; the effective read re-measures nothing."""
    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import load_effective_task_result

    source = tmp_path / "report.txt"
    source.write_text("original report")
    ctx = SimpleNamespace(drive_root=tmp_path / "data", task_id="capture")
    captured = artifacts.copy_file_to_task_artifacts(ctx, source, immutable=immutable)
    write_task_result(ctx.drive_root, ctx.task_id, "completed", artifacts=[captured])
    Path(captured["path"]).write_text("later changed report")
    assert load_effective_task_result(ctx.drive_root, ctx.task_id)["artifacts"] == [captured]
    projected = artifacts.merge_artifact_records(
        [captured], artifacts.collect_task_artifact_records(ctx.drive_root, ctx.task_id))[0]
    if immutable:
        assert projected["immutable"] is True
        assert projected["sha256"] == captured["sha256"]
        assert projected["size"] == captured["size"]
        assert projected["status"] == "failed" and projected["errors"]
        with pytest.raises(OSError, match="verification"):
            artifacts.copy_file_to_task_artifacts(ctx, Path(captured["path"]))
        registered = artifacts.registered_task_artifact(ctx.drive_root, ctx.task_id, captured["name"])
        assert registered["sha256"] == captured["sha256"]
    else:
        assert projected["sha256"] == sha256(b"later changed report").hexdigest()
        assert projected["status"] == "ready"


def test_immutable_child_copy_back_retains_name_bytes_and_original_identity(tmp_path):
    from ouroboros.headless import prepare_task_drive, copy_child_task_result
    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import load_effective_task_result

    parent = tmp_path / "canonical"
    child = prepare_task_drive(parent, "capture", "empty")
    source = child / "report.txt"
    source.write_text("original report")
    record = artifacts.copy_file_to_task_artifacts(
        SimpleNamespace(drive_root=child, task_id="capture"), source, immutable=True)
    write_task_result(child, "capture", "completed", artifacts=[record], artifact_status="ready")
    write_task_result(parent, "capture", "running", headless_child_drive_root=str(child))
    # The read lists the child's capture where it lies; copy-back alone publishes it.
    assert load_effective_task_result(parent, "capture")["artifacts"] == [record]
    assert not artifacts.task_artifact_dir_path(parent, "capture").exists()
    rebased = copy_child_task_result(parent, {"id": "capture", "drive_root": str(child)})["artifacts"][0]
    assert rebased["name"] == record["name"] and rebased["sha256"] == record["sha256"]
    assert rebased["immutable"] is True
    assert Path(rebased["path"]).parent == artifacts.task_artifact_dir_path(parent, "capture")
    assert Path(rebased["path"]).read_text() == "original report"
    Path(record["path"]).write_text("changed child bytes")
    # Neither effective reads nor physical copy-back may replace captured parent bytes.
    load_effective_task_result(parent, "capture")
    copied = copy_child_task_result(parent, {"id": "capture", "drive_root": str(child)})
    copied = retry_child_task_refs(parent, child, "capture")
    assert Path(rebased["path"]).read_text() == "original report"
    assert copied["artifacts"][0]["sha256"] == record["sha256"]
    with pytest.raises(OSError, match="verification"):
        artifacts.copy_file_to_task_artifacts(
            SimpleNamespace(drive_root=tmp_path / "other", task_id="capture"),
            record["path"], immutable=True, expected=record)


@pytest.mark.serial
@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("registered", [True, False])
def test_file_verification_does_not_run_on_the_asgi_loop(tmp_path, monkeypatch, method, registered):
    import asyncio
    import socket
    import threading
    import time
    import httpx
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Route
    from ouroboros.gateway import task_archive, tasks
    from ouroboros.task_results import write_task_result

    api_task_artifact = tasks.api_task_artifact
    source = tmp_path / 'report.txt'
    source.write_bytes(b'captured download bytes')
    data = tmp_path / 'data'
    record = artifacts.copy_file_to_task_artifacts(SimpleNamespace(drive_root=data, task_id='download'), source, immutable=True)
    write_task_result(data, "download", "completed", artifacts=[record])
    calls = []
    materialization_calls = []
    materialize = tasks.load_effective_task_result
    def observe_materialization(*args, **kwargs):
        materialization_calls.append(threading.get_ident())
        return materialize(*args, **kwargs)
    monkeypatch.setattr(tasks, "load_effective_task_result", observe_materialization)
    if not registered:
        monkeypatch.setattr(artifacts, "registered_task_artifact", lambda *args: None)
    original = artifacts.stream_artifact_file
    def observed(*args, **kwargs):
        calls.append(threading.get_ident())
        return original(*args, **kwargs)
    monkeypatch.setattr(artifacts, 'stream_artifact_file', observed)
    app = Starlette(routes=[Route('/api/tasks/{task_id}/artifacts/{name}', api_task_artifact, methods=['GET', 'HEAD'])])
    app.state.drive_root = data
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning'))
    thread = threading.Thread(target=server.run, kwargs={'sockets':[sock]}, daemon=True)
    thread.start()
    try:
        end = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < end:
            time.sleep(.01)
        assert server.started
        async def request():
            async with httpx.AsyncClient(timeout=10) as client:
                reply = await client.request(method, f'http://127.0.0.1:{sock.getsockname()[1]}/api/tasks/download/artifacts/{record["name"]}')
                if task_archive.CONFINED:
                    assert reply.status_code == 200, reply.text
                    assert reply.content == (source.read_bytes() if method == 'GET' else b'')
                else:
                    assert reply.status_code == 503, reply.text
                    if method == 'GET':
                        assert reply.json()['reason_code'] == 'artifact_unavailable'
                    else:
                        assert reply.content == b''
        asyncio.run(request())
        assert bool(materialization_calls) is (not registered)
        assert all(identity != thread.ident for identity in materialization_calls)
        if task_archive.CONFINED:
            if registered:
                assert len(calls) == 1, "registered downloads verify once per request"
            assert calls and all(identity != thread.ident for identity in calls), 'whole-file hashing ran synchronously on the ASGI event loop'
        else:
            assert calls == [], "unsupported platforms refuse before reading file bytes (#1297)"
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()


@pytest.mark.parametrize("canonical_exists", [False, True])
def test_failed_child_capture_is_explicit_and_other_files_still_publish(tmp_path, canonical_exists):
    from ouroboros.headless import prepare_task_drive, copy_child_task_result, remove_subagent_task_drive
    from ouroboros.observability import retry_pending_child_ref_promotions
    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import load_effective_task_result

    parent = tmp_path / "canonical"
    child = prepare_task_drive(parent, "capture", "empty")
    child_ctx = SimpleNamespace(drive_root=child, task_id="capture")
    records = []
    for name in ("a-report.txt", "z-neighbor.txt"):
        source = child / name
        source.write_bytes(name.encode())
        records.append(artifacts.copy_file_to_task_artifacts(child_ctx, source, immutable=True))
    axes = {"execution": {"status": "ok"}, "objective": {"status": "pass", "source": "task_acceptance_review"}}
    write_task_result(child, "capture", "completed", artifacts=records, artifact_status="ready",
                      outcome_axes=axes, accounted_upper_bound_usd=3.5, cost_final=True)
    write_task_result(parent, "capture", "running", headless_child_drive_root=str(child))
    task = {"id": "capture", "drive_root": str(child)}
    if canonical_exists:
        assert copy_child_task_result(parent, task)["artifact_bundle"]["status"] == "ready"
    bad = records[0]
    Path(bad["path"]).write_bytes(b"changed child bytes before copy")
    copied = copy_child_task_result(parent, task)
    copied = retry_child_task_refs(parent, child, task["id"])
    result = load_effective_task_result(parent, "capture")
    row = next(item for item in result["artifacts"] if item["name"] == bad["name"])
    assert result["status"] == "completed" and result["outcome_axes"] == axes
    assert result["accounted_upper_bound_usd"] == 3.5 and result["cost_final"] is True
    assert (row["name"], row["sha256"], row["size"]) == (bad["name"], bad["sha256"], bad["size"])
    neighbor = next(item for item in result["artifacts"] if item["name"] == records[1]["name"])
    assert neighbor["status"] == "ready" and Path(neighbor["path"]).read_bytes() == b"z-neighbor.txt"
    assert Path(neighbor["path"]).parent == artifacts.task_artifact_dir_path(parent, "capture")
    if canonical_exists:
        assert row["status"] == result["artifact_status"] == result["artifact_bundle"]["status"] == "ready"
        assert Path(row["path"]).read_bytes() == b"a-report.txt"
        Path(bad["path"]).unlink()
        reused = artifacts.copy_file_to_task_artifacts(
            SimpleNamespace(drive_root=parent, task_id="capture"), bad["path"], immutable=True, expected=bad)
        assert reused["path"] == row["path"] and reused["sha256"] == bad["sha256"]
    else:
        assert row["copy_status"] == "failed" and row["copy_error"]
        assert result["artifact_status"] == result["artifact_bundle"]["status"] == "missing"
        assert copied["child_ref_promotion"]["status"] == "incomplete"
        assert not remove_subagent_task_drive(parent, "capture", live=lambda _task: False)
        Path(bad["path"]).write_bytes(b"a-report.txt")
        assert retry_pending_child_ref_promotions(parent)["completed"] == ["capture"]
        recovered = load_effective_task_result(parent, "capture")
        assert recovered["artifact_status"] == recovered["artifact_bundle"]["status"] == "ready"
        assert recovered["status"] == "completed" and recovered["accounted_upper_bound_usd"] == 3.5


def test_unchanged_pending_ref_retry_rewrites_nothing_and_keeps_late_facts(tmp_path, monkeypatch):
    """#1305: retrying a still-unpromotable ref leaves bytes, mtime and updated_at alone; a
    genuine late fact still lands, survives the next retry, and a restored source converges."""
    from ouroboros.headless import prepare_task_drive, copy_child_task_result
    from ouroboros.observability import retry_pending_child_ref_promotions
    from ouroboros.task_results import task_result_path, write_task_result

    parent = tmp_path / "canonical"
    child = prepare_task_drive(parent, "stale", "empty")
    source = child / "report.txt"
    source.write_bytes(b"report")
    record = artifacts.copy_file_to_task_artifacts(
        SimpleNamespace(drive_root=child, task_id="stale"), source, immutable=True)
    write_task_result(child, "stale", "failed", artifacts=[record], artifact_status="ready")
    write_task_result(parent, "stale", "running", headless_child_drive_root=str(child))
    Path(record["path"]).write_bytes(b"appended after the digest was captured")
    assert copy_child_task_result(parent, {"id": "stale", "drive_root": str(child)})[
        "child_ref_promotion"]["status"] == "incomplete"
    # Adoption only owes history; the first archive attempt establishes its
    # inventory and failed-copy facts. Unchanged retries after that do not write.
    assert retry_pending_child_ref_promotions(parent)["pending"] == ["stale"]
    row = task_result_path(parent, "stale")
    before, stamp = row.read_bytes(), row.stat().st_mtime_ns
    report = retry_pending_child_ref_promotions(parent)
    assert report["unchanged"] == report["pending"] == ["stale"]
    assert not report["retried"] and not report["errors"]
    assert row.read_bytes() == before and row.stat().st_mtime_ns == stamp
    write_task_result(parent, "stale", "failed", accounted_upper_bound_usd=4.25)
    late = row.read_bytes()
    report = retry_pending_child_ref_promotions(parent)
    assert report["retried"] == report["pending"] == ["stale"] and not report["unchanged"]
    assert row.read_bytes() == late and json.loads(late)["accounted_upper_bound_usd"] == 4.25
    Path(record["path"]).write_bytes(b"report")
    attempts = []
    def transient_copy(*args, **kwargs):
        attempts.append(args)
        raise OSError("temporary destination write failure")
    with monkeypatch.context() as failure:
        failure.setattr(artifacts, "copy_artifact_file", transient_copy)
        for expected_attempts in (1, 2):
            report = retry_pending_child_ref_promotions(parent)
            assert report["retried"] == report["pending"] == ["stale"] and not report["unchanged"]
            assert len(attempts) == expected_attempts
    assert retry_pending_child_ref_promotions(parent)["completed"] == ["stale"]
    settled = json.loads(row.read_bytes())
    assert settled["status"] == "failed" and settled["accounted_upper_bound_usd"] == 4.25
    assert settled["child_ref_promotion"]["status"] == "complete"


@pytest.mark.parametrize("copy_failure", [False, True])
def test_failed_artifact_bundle_drives_public_and_routing_status_without_mutating_capture(tmp_path, copy_failure):
    from ouroboros.outcomes import artifact_bundle_from_result, public_task_result
    from ouroboros.server_routing_context import _task_result_ground_truth
    from ouroboros.task_status import effective_task_result

    row = {"name": "report.txt", "path": str(tmp_path / "report.txt"), "status": "failed", "errors": ["capture failed"]}
    if copy_failure:
        row.update(status="ready", copy_status="failed", copy_error="copy failed")
    result = {"task_id": "capture", "status": "completed", "artifacts": [row], "artifact_status": "ready"}
    result["artifact_bundle"] = artifact_bundle_from_result(result)
    expected = "missing" if copy_failure else "failed"
    assert result["artifact_bundle"]["status"] == expected
    assert public_task_result(result)["artifact_status"] == expected
    assert _task_result_ground_truth(result)["artifact_status"] == expected
    assert effective_task_result(tmp_path, result, materialize_artifacts=False)["artifact_status"] == expected
    assert result["artifact_status"] == "ready" and result["status"] == "completed"
    # A capture-level failure is independent of an earlier ready bundle.
    assert artifact_bundle_from_result({"artifact_status": "failed", "artifact_bundle": {"status": "ready"}})["status"] == "failed"


@pytest.mark.parametrize("drive_kind", ["headless", "direct"])
def test_first_materialization_copy_failure_keeps_each_gc_root(tmp_path, monkeypatch, drive_kind):
    import time
    from ouroboros import headless
    from ouroboros.task_results import load_task_result, write_task_result

    parent = tmp_path / "canonical"
    child = (headless.prepare_task_drive(parent, "capture", "empty") if drive_kind == "headless"
             else parent / headless.TASK_DRIVES_DIR / "capture")
    child.mkdir(parents=True, exist_ok=True)
    source = child / "report.txt"
    source.write_bytes(b"complete report")
    record = artifacts.copy_file_to_task_artifacts(
        SimpleNamespace(drive_root=child, task_id="capture"), source, immutable=True)
    write_task_result(child, "capture", "completed", artifacts=[record], artifact_status="ready")
    # An adopted row (its promotion mark set) that still names the capture at its CHILD path:
    # the settlement itself owes the canonical copy, so its copy failure is the probe.
    write_task_result(parent, "capture", "completed", artifacts=[record], artifact_status="ready",
                      headless_child_drive_root=str(child),
                      child_ref_promotion={"schema_version": 1, "status": "complete", "pending_refs": []})
    original = artifacts.copy_artifact_file
    def fail_copy(src, dst, **kwargs):
        # Settlement prepares every canonical copy in private staging under the canonical
        # root: that copy failing is the canonical copy failing.
        if Path(dst).is_relative_to(parent) and not Path(dst).is_relative_to(child):
            raise OSError("controlled canonical copy failure")
        return original(src, dst, **kwargs)
    prune = headless.prune_headless_task_drives if drive_kind == "headless" else headless.prune_task_drives
    later = time.time() + 14 * 86400
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "copy_artifact_file", fail_copy)
        refused = prune(parent, retention_days=7, now=later, live=lambda _task: False)
    assert not refused["pruned"] and child.is_dir(), refused
    assert refused["custody_pending"] == [{"task_id": "capture", "reason": "artifact_source_mismatch"}]
    assert Path(record["path"]).read_bytes() == b"complete report"
    assert load_task_result(parent, "capture")["artifacts"] == [record]  # the row still names its source
    assert not artifacts.task_artifact_dir_path(parent, "capture").exists()
    settled = prune(parent, retention_days=7, now=later, live=lambda _task: False)
    assert settled["pruned"] and not child.exists(), settled
    published = load_task_result(parent, "capture")["artifacts"][0]
    assert Path(published["path"]).read_bytes() == b"complete report" and published["sha256"] == record["sha256"]
    assert Path(published["path"]).is_relative_to(artifacts.task_artifact_dir_path(parent, "capture").resolve())


@pytest.mark.parametrize("context", ["healthy", "missing", "none", "storage_failure"])
def test_small_document_keeps_inline_delivery_when_capture_is_unavailable(tmp_path, monkeypatch, context):
    import base64
    from ouroboros.tools import core_artifacts
    from ouroboros.tools.registry import ToolContext

    source = tmp_path / "report.txt"
    source.write_bytes(b"complete report")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data", task_id="send", current_chat_id=1)
    if context == "missing":
        ctx = SimpleNamespace(current_chat_id=1, pending_events=[], task_id="send", task_metadata={})
    elif context == "none":
        ctx.drive_root = None
    elif context == "storage_failure":
        def unavailable(*args, **kwargs):
            raise OSError("controlled artifact-store failure")
        monkeypatch.setattr(artifacts, "copy_file_to_task_artifacts", unavailable)
    assert core_artifacts._send_file(ctx, str(source)).startswith("OK")
    event, = ctx.pending_events
    assert base64.b64decode(event["file_base64"]) == source.read_bytes()
    if context != "healthy":
        assert event["file_ref"] is None
        assert event["download_url"] == event["download_url_compat"] == ""
    from supervisor import events_chat_delivery, message_bus

    frames, host_events, errors = [], [], []
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(message_bus, "_advance_project_visible_revision", lambda *args: None)
    monkeypatch.setattr(message_bus, "publish_event", lambda kind, payload: host_events.append(payload))
    monkeypatch.setattr(events_chat_delivery, "_bound_project_chat_id", lambda *args: None)
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = frames.append
    events_chat_delivery._handle_send_document(event, SimpleNamespace(
        DRIVE_ROOT=tmp_path / "data", bridge=bridge,
        append_jsonl=lambda path, row: errors.append(row),
    ))
    assert errors == []
    assert len(frames) == len(host_events) == 1
    assert frames[0]["file_base64"] == host_events[0]["file_base64"] == event["file_base64"]


@pytest.mark.parametrize("failure", ["large_uncaptured", "source_unreadable", "capture_refused"])
def test_inline_fallback_keeps_the_source_and_capture_boundaries(tmp_path, monkeypatch, failure):
    from ouroboros.tools import core_artifacts
    from ouroboros.tools.registry import ToolContext

    source = tmp_path / "report.txt"
    source.write_bytes(b"complete report")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data", task_id="send", current_chat_id=1)
    def unavailable(*args, **kwargs):
        if failure == "capture_refused":
            return None
        raise OSError("controlled artifact-store failure")
    monkeypatch.setattr(artifacts, "copy_file_to_task_artifacts", unavailable)
    if failure == "large_uncaptured":
        monkeypatch.setattr(core_artifacts, "_MAX_DOCUMENT_FILE_BYTES", 4)
    elif failure == "source_unreadable":
        original = Path.open
        def refuse_source(path, *args, **kwargs):
            if path == source:
                raise PermissionError("controlled source permission refusal")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "open", refuse_source)
    assert not core_artifacts._send_file(ctx, str(source)).startswith("OK")
    assert ctx.pending_events == []


@pytest.mark.parametrize("initial_input", [False, True])
@pytest.mark.parametrize("count,failure", [(1, ""), (28, ""), (28, "file"), (28, "manifest"), (28, "mailbox")])
def test_acknowledged_owner_inputs_survive_copyback_retry_mailbox_cleanup_and_gc(
    tmp_path, monkeypatch, initial_input, count, failure,
):
    import time
    from ouroboros import headless, observability, owner_mailbox
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.terminal_delivery import cleanup_settled_owner_mailbox

    parent, task_id = tmp_path / "canonical", "owner-inputs"
    child = headless.prepare_task_drive(parent, task_id, "empty")
    initial = []
    if initial_input:
        source = tmp_path / "initial.txt"
        source.write_bytes(b"initial input")
        initial = artifacts.stage_task_attachments(child, task_id, [{"path": str(source)}])
    contract = artifacts.attachment_manifest_projection(child, task_id, initial)
    sources = []
    for index in range(count):
        source = tmp_path / f"late-{index}.txt"
        source.write_bytes(f"late input {index}".encode())
        sources.append({"path": str(source)})
    rows = artifacts.stage_task_attachments(child, task_id, sources)
    assert owner_mailbox.write_owner_message(
        child, "Use these additional inputs", task_id, msg_id="owner-more", attachment_manifest=rows,
    )
    mailbox = owner_mailbox._mailbox_path(child, task_id)
    entry = json.loads(mailbox.read_text(encoding="utf-8"))
    ref = entry.get("attachment_manifest_ref")
    full_body = artifacts.read_actor_source_bytes(child, task_id, ref) if ref else None
    assert owner_mailbox.acknowledge_task_messages(child, task_id, ["owner-more"], wake_id="test")
    assert owner_mailbox.drain_owner_entries(child, task_id) == []
    write_task_result(child, task_id, "completed", result="done", task_contract=contract, artifact_status="ready")
    task = {"id": task_id, "drive_root": str(child)}
    original_copy = artifacts.copy_artifact_file

    def unavailable_copy(source, destination, **kwargs):
        path = Path(source)
        if ((failure == "file" and path.name == "late-0.txt")
                or (failure == "manifest" and path.suffix == ".json")):
            raise OSError("controlled owner input copy failure")
        return original_copy(source, destination, **kwargs)

    with monkeypatch.context() as patch:
        if failure in {"file", "manifest"}:
            patch.setattr(artifacts, "copy_artifact_file", unavailable_copy)
        elif failure == "mailbox":
            def unreadable(*args, **kwargs):
                raise PermissionError("controlled mailbox read failure")
            patch.setattr(owner_mailbox, "owner_attachment_manifest", unreadable)
        result = headless.copy_child_task_result(parent, task)
        result = retry_child_task_refs(parent, child, task["id"])
    if failure:
        assert result["child_ref_promotion"]["status"] == "incomplete"
        assert any(row["path"] == str(mailbox) for row in result["child_ref_promotion"]["pending_refs"])
        cleanup_settled_owner_mailbox(parent, task_id, task)
        assert mailbox.is_file()
        assert not headless.remove_subagent_task_drive(parent, task_id, live=lambda _task: False)
        assert observability.retry_pending_child_ref_promotions(parent)["completed"] == [task_id]
        result = load_task_result(parent, task_id)
        assert not mailbox.exists(), "successful retry releases retained mail through its existing owner"
    else:
        cleanup_settled_owner_mailbox(parent, task_id, task)
        assert not mailbox.exists()
    assert result["child_ref_promotion"]["status"] == "complete"
    assert result["child_ref_promotion"]["pending_refs"] == []
    assert len(artifacts.resolve_attachment_manifest(parent, task_id, result["task_contract"])) == len(initial)
    gc = headless.prune_headless_task_drives(parent, retention_days=1, now=time.time() + 90 * 86400, live=lambda _task: False)
    assert [row["task_id"] for row in gc["pruned"]] == [task_id]
    assert not child.exists()
    if ref:
        assert artifacts.read_actor_source_bytes(parent, task_id, ref) == full_body
        assert len(artifacts.resolve_attachment_manifest(parent, task_id, entry)) == count
    for row in rows:
        captured = artifacts.task_artifact_dir_path(parent, task_id) / row["relpath"]
        assert artifacts.stream_artifact_file(captured, expected=row)["sha256"] == row["sha256"]
    assert artifacts.collect_task_artifact_records(parent, task_id) == []
