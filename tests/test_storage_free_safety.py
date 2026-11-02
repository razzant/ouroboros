"""What the free-space owners must never lose or write: a plain copy's unsaved work, a copy whose
record cannot be read, and any byte during a preview."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from ouroboros import subagent_worktrees as sw
from ouroboros.headless import task_artifacts_dir
from ouroboros.workspace_patch_capture import write_workspace_patch_artifacts
from tests._store_journal_shared import tree_state
from tests._store_shared import seed_result
from tests.test_body_candidate_retention import candidate  # noqa: F401 -- the body fixture
from tests.test_storage_free import _fake_sizer, _group, _init_repo, _owner, _preview, _run
from tests.test_store import document


@pytest.fixture
def copies(tmp_path, monkeypatch):
    root, repo, wtroot = tmp_path / "data", tmp_path / "repo", tmp_path / "wtroot"
    _init_repo(repo)
    root.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(wtroot))
    _fake_sizer(monkeypatch)
    return root, repo, wtroot


def _copy(env, task_id="finished"):
    root, repo, wtroot = env
    handle = sw.provision_worktree(repo_dir=repo, task_id=task_id, worktree_root=wtroot, data_dir=root)
    seed_result(root, task_id, document(task_id, "completed"))
    return handle


def _capture(env, handle, task_id="finished"):
    """The real producer, as the task's end runs it."""
    task = {"id": task_id, "task_constraint": {
        "mode": "acting_subagent", "surface": "self_worktree", "base_sha": handle.base_sha,
        "write_root": handle.path, "parent_only_commit": True, "return_kind": "workspace_patch"}}
    _, manifest = write_workspace_patch_artifacts(Path(handle.path), task_artifacts_dir(env[0], task_id), task=task)
    return manifest


def _prune(env):
    return sw.prune_orphans(data_dir=env[0], worktree_root=env[2], retention_days=0, live=lambda _task: False)


def _git(path, *args):
    return subprocess.run(["git", *args], cwd=str(path), check=True, capture_output=True, text=True).stdout


def test_a_copy_with_nothing_unique_goes_without_any_capture(copies):
    handle = _copy(copies)
    assert _prune(copies)["removed"] == 1 and not Path(handle.path).exists()


@pytest.mark.parametrize("rel", [".venv/bin/python", ".cache/inputs/held.txt", "__pycache__/a.pyc", "scratch.tmp"])
def test_a_name_the_patch_rules_leave_out_proves_nothing_reproducible(copies, rel):
    """Those rules decide transport; no producer recorded these as reproducible, so the copy stays."""
    handle = _copy(copies)
    target = Path(handle.path) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x")
    assert _prune(copies)["removed"] == 0 and target.read_bytes() == b"x"


def test_a_late_write_after_the_first_look_keeps_the_copy(copies):
    """The lock-time decision: the checkout changed after the pass's first judgement under its lock."""
    handle = _copy(copies)
    late = Path(handle.path) / "late.txt"
    calls = []

    def live(_task):
        if not calls:
            late.write_text("arrived during the pass\n", encoding="utf-8")
        calls.append(1)
        return False

    report = sw.prune_orphans(data_dir=copies[0], worktree_root=copies[2], retention_days=0, live=live)
    assert report["removed"] == 0 and late.read_text(encoding="utf-8") == "arrived during the pass\n"
    assert len(calls) >= 2, "the task's end was asked again at the destructive boundary"


def test_a_task_alive_again_at_the_destructive_boundary_keeps_the_copy(copies):
    handle = _copy(copies)
    answers = iter([False])
    report = sw.prune_orphans(data_dir=copies[0], worktree_root=copies[2], retention_days=0,
                              live=lambda _task: next(answers, True))
    assert report["removed"] == 0 and Path(handle.path).is_dir()


def test_an_unreadable_branch_is_kept_though_git_said_no_ref(copies, monkeypatch):
    from ouroboros import copy_capture_proof as proof

    handle = _copy(copies)
    row = sw._record_at(handle.path, copies[0])
    sw._force_rmtree(Path(handle.path))
    real = sw._git

    def broken(repo, *args, **kwargs):
        done = real(repo, *args, **kwargs)
        return subprocess.CompletedProcess(done.args, 128, b"", b"") if args[0] == "rev-parse" else done

    monkeypatch.setattr(sw, "_git", broken)
    assert proof.plain_copy_verdict(row, copies[0]) == {"remove": True, "keep_branch": True, "reason": "checkout_missing"}


def test_a_copy_whose_capture_failed_stays_whole(copies):
    handle = _copy(copies)
    (Path(handle.path) / "work.txt").write_text("only here\n", encoding="utf-8")
    artifacts = task_artifacts_dir(copies[0], "finished")
    (artifacts / "workspace_patch.json").write_text(json.dumps(
        {"status": "failed", "workspace_root": handle.path, "errors": [{"type": "exception"}]}), encoding="utf-8")
    assert _prune(copies)["removed"] == 0
    assert (Path(handle.path) / "work.txt").read_text(encoding="utf-8") == "only here\n"
    assert sw._record_at(handle.path, copies[0]) is not None


def test_a_copy_whose_capture_never_ran_stays_whole(copies):
    handle = _copy(copies)
    (Path(handle.path) / "work.txt").write_text("only here\n", encoding="utf-8")
    assert _prune(copies)["removed"] == 0 and (Path(handle.path) / "work.txt").is_file()


def test_a_commit_the_task_made_keeps_its_copy_and_branch_though_the_capture_saw_none(copies):
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "work.txt").write_text("committed\n", encoding="utf-8")
    _git(path, "add", "work.txt")
    _git(path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "work", "--no-verify")
    _capture(copies, handle)  # a moved HEAD fails this capture; its commit lives only here
    assert _prune(copies)["removed"] == 0 and (path / "work.txt").is_file()
    assert _git(copies[1], "rev-parse", "--verify", f"refs/heads/{handle.branch}").strip()


def test_a_file_changed_after_a_good_capture_stays(copies):
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "work.txt").write_text("saved\n", encoding="utf-8")
    assert _capture(copies, handle)["status"] == "ready_with_changes"
    stamp = os.stat(path / "work.txt")
    (path / "work.txt").write_text("later\n", encoding="utf-8")
    os.utime(path / "work.txt", (stamp.st_atime, stamp.st_mtime))  # the clock cannot be what proves it
    assert _prune(copies)["removed"] == 0 and (path / "work.txt").read_text(encoding="utf-8") == "later\n"


def test_a_mode_changed_after_a_good_capture_stays(copies):
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "tool.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    assert _capture(copies, handle)["status"] == "ready_with_changes"
    (path / "tool.sh").chmod((path / "tool.sh").stat().st_mode | stat.S_IXUSR)
    assert _prune(copies)["removed"] == 0 and (path / "tool.sh").is_file()


def test_a_saved_patch_whose_bytes_changed_proves_nothing(copies):
    handle = _copy(copies)
    (Path(handle.path) / "work.txt").write_text("saved\n", encoding="utf-8")
    assert _capture(copies, handle)["status"] == "ready_with_changes"
    patch = task_artifacts_dir(copies[0], "finished") / "workspace.patch"
    patch.write_bytes(patch.read_bytes() + b"\n")
    assert _prune(copies)["removed"] == 0 and (Path(handle.path) / "work.txt").is_file()


def test_ignored_and_oversized_files_the_patch_left_out_keep_the_copy(copies):
    handle = _copy(copies)
    path = Path(handle.path)
    (path / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    (path / "secret.txt").write_text("only here\n", encoding="utf-8")
    assert _capture(copies, handle)["status"] == "ready_with_changes"
    assert _prune(copies)["removed"] == 0 and (path / "secret.txt").is_file()


def test_a_good_capture_lets_text_links_modes_and_an_edited_binary_go(copies):
    (copies[1] / "tracked.bin").write_bytes(bytes(range(256)) * 8)  # the copy's baseline carries it
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "tracked.bin").write_bytes(bytes(reversed(range(256))) * 8)
    (path / "note.txt").write_text("new\n", encoding="utf-8")
    (path / "tool.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (path / "tool.sh").chmod(0o755)
    (path / "link").symlink_to("note.txt")
    assert _capture(copies, handle)["status"] == "ready_with_changes"
    assert _prune(copies)["removed"] == 1 and not path.exists()
    assert sw._record_at(handle.path, copies[0]) is None


def _big(size=5 * 1024 * 1024 + 1):
    return (b"long line of generated text\n" * (size // 28 + 1))[:size]


def _restored(env, rel):
    """The saved bytes as the integration owner would restore them, from the task's capture alone."""
    import zipfile

    from ouroboros.workspace_file_outputs import file_output_changes

    artifacts = task_artifacts_dir(env[0], "finished")
    manifest = json.loads((artifacts / "workspace_patch.json").read_text(encoding="utf-8"))
    row = next(row for row in file_output_changes(manifest, artifacts) if row["path"] == rel)
    with zipfile.ZipFile(row["archive_path"]) as archive:
        return archive.read(row["member"])


@pytest.mark.parametrize("beside", [True, False], ids=["with_a_patch", "patchless"])
def test_an_untracked_binary_its_verified_file_result_holds_lets_the_copy_go(copies, beside):
    handle = _copy(copies)
    path = Path(handle.path)
    blob = bytes(range(256)) * 8
    (path / "blob.bin").write_bytes(blob)
    if beside:
        (path / "note.txt").write_text("new\n", encoding="utf-8")
    manifest = _capture(copies, handle)
    assert manifest["status"] == "ready_with_changes" and "blob.bin" not in manifest["untracked_included"]
    assert bool(manifest["patch_size"]) is beside
    assert _prune(copies)["removed"] == 1 and not path.exists()
    assert _restored(copies, "blob.bin") == blob


def test_a_large_untracked_file_its_verified_file_result_holds_lets_the_copy_go(copies):
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "generated.txt").write_bytes(_big())
    assert _capture(copies, handle)["file_outputs"]
    assert _prune(copies)["removed"] == 1 and not path.exists()
    assert _restored(copies, "generated.txt") == _big()


def test_a_large_tracked_edit_and_deletion_held_by_file_results_let_the_copy_go(copies, monkeypatch):
    import ouroboros.workspace_patch_capture as capture

    monkeypatch.setattr(capture, "_PATCH_FILE_REFERENCE_BYTES", 1024)
    (copies[1] / "data.bin").write_bytes(b"a" * 4096)
    (copies[1] / "gone.bin").write_bytes(b"b" * 4096)
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "data.bin").write_bytes(b"c" * 4096)
    (path / "gone.bin").unlink()
    manifest = _capture(copies, handle)
    assert {row["path"] for row in manifest["file_output_changes"]} == {"data.bin", "gone.bin"}
    assert _prune(copies)["removed"] == 1 and not path.exists()


def test_a_file_result_edited_after_its_capture_keeps_the_copy(copies):
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "blob.bin").write_bytes(bytes(range(256)) * 8)
    _capture(copies, handle)
    (path / "blob.bin").write_bytes(bytes(reversed(range(256))) * 8)
    assert _prune(copies)["removed"] == 0 and (path / "blob.bin").is_file()


def test_a_file_result_whose_mode_changed_after_its_capture_keeps_the_copy(copies):
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "blob.bin").write_bytes(bytes(range(256)) * 8)
    _capture(copies, handle)
    (path / "blob.bin").chmod(0o755)
    assert _prune(copies)["removed"] == 0 and (path / "blob.bin").is_file()


@pytest.mark.parametrize("damage", ["archive_bytes", "archive_gone", "ledger_gone", "member_lies"])
def test_a_file_result_that_cannot_be_read_back_whole_keeps_the_copy(copies, damage):
    import zipfile

    handle = _copy(copies)
    path = Path(handle.path)
    (path / "blob.bin").write_bytes(bytes(range(256)) * 8)
    manifest = _capture(copies, handle)
    artifacts = task_artifacts_dir(copies[0], "finished")
    archive = next(artifacts / row["name"] for row in manifest["file_outputs"] if row["name"].endswith(".zip"))
    ledger = next(artifacts / row["name"] for row in manifest["file_outputs"] if row["name"].endswith(".json"))
    if damage == "archive_bytes":
        raw = bytearray(archive.read_bytes())
        raw[len(raw) // 2] ^= 0xff
        archive.write_bytes(raw)
    elif damage == "archive_gone":
        archive.unlink()
    elif damage == "ledger_gone":
        ledger.unlink()
    else:  # a whole, identified archive whose member is not the recorded bytes
        with zipfile.ZipFile(archive, "w") as rewritten:
            rewritten.writestr("blob.bin", b"something else")
        for row in manifest["file_outputs"]:
            if row["name"] == archive.name:
                row["sha256"], row["size"] = _artifact_identity(archive)
        (artifacts / "workspace_patch.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _prune(copies)["removed"] == 0 and (path / "blob.bin").is_file()


def _artifact_identity(path):
    from ouroboros.artifacts import stream_artifact_file

    measured = stream_artifact_file(path)
    return measured["sha256"], measured["size"]


@pytest.mark.parametrize("rel", [".venv/bin/python", ".env", "ignored.log", "cache/__pycache__/a.pyc", "scratch.tmp"])
def test_what_no_capture_holds_keeps_the_copy_beside_a_saved_file_result(copies, rel):
    """Ignored, environment and credential-shaped files really are nowhere else: that stays the owner's choice."""
    (copies[1] / ".gitignore").write_text("*.log\n", encoding="utf-8")
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "blob.bin").write_bytes(bytes(range(256)) * 8)
    target = path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"only here")
    _capture(copies, handle)
    assert _prune(copies)["removed"] == 0 and target.read_bytes() == b"only here"


def _counted_hashing(monkeypatch):
    calls, real = [], sw._git_env

    def counted(repo, *args, **kwargs):
        if args[0] == "hash-object":
            calls.append(args)
        return real(repo, *args, **kwargs)

    monkeypatch.setattr(sw, "_git_env", counted)
    return calls


AWKWARD = ['with\nnewline.txt', 'quo"te.txt', '"leading.txt', 'back\\slash.txt', 'tab\t.txt', 'space d.txt', 'ю.txt']


def test_one_hashing_process_proves_every_saved_file_whatever_its_name(copies, monkeypatch):
    handle = _copy(copies)
    path = Path(handle.path)
    for number in range(30):
        (path / f"note{number}.txt").write_text(f"{number}\n", encoding="utf-8")
    for name in AWKWARD:
        (path / name).write_text(f"{name}\n", encoding="utf-8")
    assert _capture(copies, handle)["status"] == "ready_with_changes"
    calls = _counted_hashing(monkeypatch)
    assert _prune(copies)["removed"] == 1 and not path.exists()
    assert len(calls) == 1, calls


@pytest.mark.parametrize("name", AWKWARD)
def test_an_awkwardly_named_file_edited_after_its_capture_still_keeps_the_copy(copies, name):
    handle = _copy(copies)
    path = Path(handle.path)
    (path / name).write_text("saved\n", encoding="utf-8")
    (path / "other.txt").write_text("saved\n", encoding="utf-8")
    _capture(copies, handle)
    (path / name).write_text("edited\n", encoding="utf-8")
    assert _prune(copies)["removed"] == 0 and (path / name).read_text(encoding="utf-8") == "edited\n"


def test_a_gone_checkout_goes_but_a_branch_with_commits_stays(copies):
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "work.txt").write_text("committed\n", encoding="utf-8")
    _git(path, "add", "work.txt")
    _git(path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "work", "--no-verify")
    sw._force_rmtree(path)
    assert _prune(copies)["removed"] == 1 and sw._record_at(handle.path, copies[0]) is None
    assert _git(copies[1], "rev-parse", "--verify", f"refs/heads/{handle.branch}").strip()


def test_a_gone_checkout_with_nothing_on_its_branch_takes_the_branch_along(copies):
    handle = _copy(copies)
    sw._force_rmtree(Path(handle.path))
    assert _prune(copies)["removed"] == 1
    assert handle.branch not in _git(copies[1], "branch", "--list")


def test_the_preview_keeps_an_unsaved_copy_out_of_the_free_groups_and_the_run_keeps_it(copies):
    handle, clean = _copy(copies), _copy(copies, "clean")
    (Path(handle.path) / "work.txt").write_text("only here\n", encoding="utf-8")
    owner = _owner({"finished": False, "clean": False})
    view = _preview(owner, copies[0], copies[0].parent)
    group = _group(view, "working_copies")
    assert [item["name"] for item in group["items"]] != [] and group["count"] == 1 and group["in_use"] == 1
    assert _run(owner, copies[0], copies[0].parent, groups=("working_copies",))["run"]["removed"] == 1
    assert Path(handle.path).is_dir() and not Path(clean.path).exists()


# ---- unreadable records and a pure preview ---------------------------------------------------


def _unreadable_record_copy(env, task_id="finished"):
    handle = _copy(env, task_id)
    sw._record_file(handle.path, env[0]).write_text("{torn", encoding="utf-8")
    return handle


def test_a_folder_whose_own_record_is_unreadable_is_no_lost_copy(copies):
    handle = _unreadable_record_copy(copies)
    owner = _owner({"finished": False})
    view = _preview(owner, copies[0], copies[0].parent)
    lost = _group(view, "lost_copies")
    assert lost["count"] == 0 and lost["unreadable"] == 1
    assert Path(handle.path).is_dir()


def test_a_run_rechecks_a_record_torn_after_its_preview(copies):
    handle = copies[2] / "dlg_lost-task_0123456789abcdef"
    handle.mkdir(parents=True)
    owner = _owner({"lost-task": False})
    assert _group(_preview(owner, copies[0], copies[0].parent), "lost_copies")["count"] == 1
    sw._record_file(handle, copies[0]).parent.mkdir(parents=True, exist_ok=True)
    sw._record_file(handle, copies[0]).write_text("{torn", encoding="utf-8")
    view = _run(owner, copies[0], copies[0].parent, groups=("lost_copies",))
    assert view["run"]["removed"] == 0 and handle.is_dir()


def test_an_unlistable_records_folder_offers_no_lost_copy(copies, monkeypatch):
    folder = copies[2] / "dlg_lost-task_0123456789abcdef"
    folder.mkdir(parents=True)
    records = sw._records_dir(copies[0])
    records.mkdir(parents=True)
    real = os.listdir
    monkeypatch.setattr(os, "listdir", lambda p=".": (_ for _ in ()).throw(PermissionError(13, "denied"))
                        if Path(p) == records else real(p))
    view = _preview(_owner({"lost-task": False}), copies[0], copies[0].parent)
    assert _group(view, "lost_copies")["count"] == 0 and _group(view, "lost_copies")["unreadable"] == 1


def test_registered_checkout_never_reads_an_unreadable_record_as_no_record(copies):
    from ouroboros.private_copy_read import checkout_registration

    handle = _unreadable_record_copy(copies)
    assert checkout_registration(handle.path, copies[0]) == "unknown"
    assert sw.registered_checkout(handle.path, copies[0]) is True
    assert checkout_registration(copies[2] / "elsewhere", copies[0]) == "absent"
    assert sw.registered_checkout(copies[2] / "elsewhere", copies[0]) is False


def test_the_preview_imports_nothing_and_reports_nothing(copies):
    root, _repo, wtroot = copies
    _unreadable_record_copy(copies)  # the provisioning itself imports: the legacy file arrives after it
    known = wtroot / "known"
    known.mkdir(parents=True)
    legacy = sw._legacy_registry_path(root)
    legacy.write_text(json.dumps({"worktrees": [{"path": str(known), "task_id": "known", "branch": ""}]}), encoding="utf-8")
    sw._imported_roots.discard(str(root))
    sw._reported_unreadable.clear()
    before = tree_state(root, mtimes=True)
    before_legacy = legacy.read_bytes()
    view = _preview(_owner({"known": False, "finished": False}), root, root.parent)
    assert tree_state(root, mtimes=True) == before
    assert legacy.read_bytes() == before_legacy and not (root / "logs" / "events.jsonl").exists()
    assert not any(sw._records_dir(root).glob(".legacy-import*"))
    assert _group(view, "lost_copies")["count"] == 0, "a row of the legacy file still names its folder"

    from ouroboros.private_copy_read import checkout_registration

    assert checkout_registration(known, root) == "registered" and legacy.is_file()


def test_a_malformed_legacy_registry_stays_and_its_folders_stay_offered_as_unregistered(copies):
    root, _repo, wtroot = copies
    folder = wtroot / "dlg_old-task_0123456789abcdef"
    folder.mkdir(parents=True)
    legacy = sw._legacy_registry_path(root)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("{torn", encoding="utf-8")
    sw._imported_roots.discard(str(root))
    view = _preview(_owner({"old-task": False}), root, root.parent)
    assert _group(view, "lost_copies")["count"] == 1 and legacy.read_text(encoding="utf-8") == "{torn"


# ---- a body candidate: a name the patch rules leave out proves nothing either ---------------


@pytest.mark.parametrize("rel", [".cache/inputs/held.txt", ".venv/owner_notes.txt", "scratch.tmp"])
def test_a_body_candidate_keeps_what_only_a_name_called_disposable(candidate, rel):  # noqa: F811
    ctx, data, scratch = candidate
    target = Path(ctx.repo_dir) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("unique sentinel\n", encoding="utf-8")
    assert sw.prune_orphans(retention_days=0) == {"removed": 0, "kept": 1}
    assert target.read_text(encoding="utf-8") == "unique sentinel\n" and scratch.is_dir()
    target.unlink()
    assert sw.prune_orphans(retention_days=0) == {"removed": 1, "kept": 0}, "the allowed path, once nothing is left"
    assert not Path(ctx.repo_dir).exists() and not scratch.exists()


def test_a_clean_body_candidate_and_its_declared_scratch_still_go(candidate):  # noqa: F811
    ctx, data, scratch = candidate
    (scratch / "tmp" / "venv").mkdir(parents=True)
    (scratch / "tmp" / "venv" / "installed").write_bytes(b"disposable")
    assert sw.prune_orphans(retention_days=0) == {"removed": 1, "kept": 0}
    assert not Path(ctx.repo_dir).exists() and not scratch.exists()


def test_a_name_excluded_file_arriving_after_the_verdict_keeps_the_candidate(candidate, monkeypatch):  # noqa: F811
    from ouroboros import body_candidate as body

    ctx, data, _scratch = candidate
    late = Path(ctx.repo_dir) / ".cache" / "late.txt"
    real = body.retention_verdict

    def verdict(*args, **kwargs):
        answer = real(*args, **kwargs)
        late.parent.mkdir(parents=True, exist_ok=True)
        late.write_text("arrived after the verdict\n", encoding="utf-8")
        return answer

    monkeypatch.setattr(body, "retention_verdict", verdict)
    assert sw.prune_orphans(retention_days=0)["removed"] == 0 and late.is_file()


# ---- a saved file's bytes, not its Git form --------------------------------------------------


def test_a_file_whose_bytes_differ_from_its_clean_form_stays(copies):
    (copies[1] / ".gitattributes").write_text("*.txt text eol=lf\n", encoding="utf-8")  # the copy's baseline carries it
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "crlf.txt").write_bytes(b"one\r\ntwo\r\n")
    assert _capture(copies, handle)["status"] == "ready_with_changes"
    assert _prune(copies)["removed"] == 0 and (path / "crlf.txt").read_bytes() == b"one\r\ntwo\r\n"


def test_a_file_whose_bytes_equal_its_clean_form_goes_under_the_same_attributes(copies):
    (copies[1] / ".gitattributes").write_text("*.txt text eol=lf\n", encoding="utf-8")
    handle = _copy(copies)
    (Path(handle.path) / "lf.txt").write_bytes(b"one\ntwo\n")
    assert _capture(copies, handle)["status"] == "ready_with_changes"
    assert _prune(copies)["removed"] == 1
