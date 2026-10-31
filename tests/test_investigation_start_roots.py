"""Default-access investigations from a Git subdirectory or a folderless Project (#1609).

A non-acting session keeps its configured access as an evidence-only investigation
in a private copy. Two starting folders schedule_subagent hands such children used
to die before the start: a Git SUBDIRECTORY (the investigation copies the enclosing
worktree as a linked worktree and starts in the matching folder) and a folderless Project task's scratch
(the existing input owner stages this invocation's inputs and the engine copies that
host-prepared folder, never the live scratch). Explicit readonly keeps reading in
place. Every start goes through the registered ``delegate_start`` handler with the
configured session row; only the engine is a double.
"""
from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.delegate_registration_policy import INVESTIGATION_AUTHORITY
from ouroboros.delegate_start_instructions import INVESTIGATION_INSTRUCTION, scratch_fact
from ouroboros.unindexed import COPIES
from ouroboros.workspace_preflight import SESSION_SCRATCH_DIRNAME, SNAPSHOT_ENVIRONMENT_NOTE

ENGINE_VERSION = "3.25.1"  # carries workspaceRoot (3.8.1+) and scope.ephemeral (3.3.7+)


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True).stdout


class Engine:
    """Route health, registration, start and one directory work product."""

    engine_version, engine_build_sha = ENGINE_VERSION, "build-1"

    def __init__(self):
        self.starts, self.grants, self.registered, self.decisions = [], [], [], []
        self.body = b"reproduced with the staged inputs\n"
        self.raw = b""

    def handshake(self, **_kw): return {}
    def agent_capabilities(self):
        return {"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                               "accessProfilesSupported": ["readonly", "workspace_write", "full"]}],
                "mutability": {"workspaceKinds": ["git", "directory"]}, "runControlKeys": ["continueFrom"]}
    def quota_snapshots(self): return []
    def find_project_id(self, root): return ""
    def project_registration(self, root):
        self.registered.append(root)
        return "prj-1", True
    def register_project(self, root): return self.project_registration(root)[0]
    def ensure_full_access(self, root): self.grants.append(root)
    def close(self): pass

    def start_run(self, request, *, idempotency_key=""):
        self.starts.append(request)
        state = {"kind": "file", "sha256": "sha256:" + sha256(self.body).hexdigest(), "sizeBytes": len(self.body),
                 "mode": 420, "artifactPath": "final/files/notes.txt"}
        self.raw = json.dumps({"version": 1, "sourceRoot": request["scope"]["root"], "executionRoot": "/engine/copy",
                               "isolation": "envelope", "scopePaths": ["."], "complete": True,
                               "entries": [{"path": ".review-evidence/notes.txt", "before": None, "after": state}]}).encode()
        return {"runId": f"inv-run-{len(self.starts)}", "runDir": "/engine/run"}

    def get_run(self, rid, **_kw):
        return {"summary": {"state": "succeeded", "effectiveAccess": self.starts[-1]["access"]},
                "workProduct": {"kind": "files", "files": {"manifest": "final/files/manifest.json"},
                                "meta": {"manifest_sha256": "sha256:" + sha256(self.raw).hexdigest(),
                                         "apply_state": "not_applied"}}}

    def stream_run_artifact(self, rid, path, sink, *, expected=None):
        body = self.raw if path.endswith("manifest.json") else self.body
        sink.write(body)
        return {"sha256": sha256(body).hexdigest(), "size": len(body)}

    def decide_run(self, rid, request, *, idempotency_key):
        self.decisions.append(request)
        return {"accepted": True, "status": "discarded"}


def _consumer(tmp_path, monkeypatch, access, *, workspace=None, metadata=None):
    """A non-acting child whose configured session row captured ``access``."""
    from ouroboros import claudexor_daemon, config
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools import delegate
    from ouroboros.tools.registry import ToolContext

    data, system = tmp_path / "data", tmp_path / "system"
    data.mkdir(exist_ok=True)
    system.mkdir(exist_ok=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snaps"))
    row = {"subagent_id": "checker", "name": "Checker", "recommended_use": "Investigation start roots",
           "access": access, "effort": "low", "route": {"kind": "agent_session", "target_id": "some-route=weak-model"}}
    monkeypatch.setattr(config, "runtime_settings", lambda: {
        "OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [row]})})
    engine = Engine()
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: engine)
    ctx = ToolContext(repo_dir=system, drive_root=data, task_constraint=TaskConstraint(mode="local_readonly_subagent"))
    if workspace is not None:
        ctx.workspace_root, ctx.workspace_mode = workspace, "external"
    ctx.task_id = "child"
    ctx.task_metadata = {"root_task_id": "parent", "parent_task_id": "parent", **(metadata or {})}
    from ouroboros.task_results import load_task_result, write_task_result
    if load_task_result(data, "parent") is None:  # the child's root exists before it delegates
        write_task_result(data, "parent", "running", root_task_id="parent")
    handler = next(tool.handler for tool in delegate.get_tools() if tool.name == "delegate_start")
    custody._CUSTODY.clear()
    return ctx, engine, lambda **kw: json.loads(handler(ctx, prompt="Check the component", subagent_id="checker", **kw))


def _project(tmp_path):
    """A worktree whose checking inputs live above the requested starting folder."""
    root = tmp_path / "project"
    folder = root / "pkg" / "component"
    folder.mkdir(parents=True)
    (root / "conftest.py").write_bytes(b"ROOT_FIXTURE = 1\n")  # byte-exact on every platform
    (folder / "mod.py").write_bytes(b"VALUE = 1\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed")
    (folder / "mod.py").write_bytes(b"VALUE = 2\n")         # unstaged checking input
    (folder / "draft.txt").write_bytes(b"untracked input\n")  # eligible untracked input
    return root, folder


def _live_state(root):
    return (_git(root, "rev-parse", "HEAD"), _git(root, "status", "--porcelain", "--untracked-files=all"),
            {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*"))
             if p.is_file() and ".git" not in p.relative_to(root).parts})


@pytest.mark.parametrize("access", ["full", "workspace_write"])
def test_git_subdirectory_investigation_copies_the_worktree_and_starts_in_the_folder(tmp_path, monkeypatch, access):
    from ouroboros.subagent_worktrees import find_execution_snapshot
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    project, folder = _project(tmp_path)
    before = _live_state(project)
    ctx, engine, start = _consumer(tmp_path, monkeypatch, access, workspace=folder)
    assert ctx.active_repo_dir() == folder
    result = start()
    assert result["status"] == "started", result
    [request] = engine.starts
    held = custody.replay(ctx.drive_root)[result["run_id"]]
    snapshot = find_execution_snapshot(held.snapshot_id)
    clone = Path(snapshot["path"])
    start_folder = clone / "pkg" / "component"
    # Configured rights, private evidence-only copy, requested starting folder.
    assert (request["access"], request["mode"]) == (access, "agent")
    assert request["execution"] == {"isolation": "live", "delegated": True, "workspaceRoot": str(start_folder)}
    assert request["scope"] == {"kind": "project", "root": str(folder)}
    assert engine.grants == ([str(folder)] if access == "full" else [])
    assert INVESTIGATION_INSTRUCTION in request["instructions"]
    assert f"the private execution root for this run is {start_folder}:" in request["instructions"]
    assert f"evidence worth returning under {start_folder / '.review-evidence'}:" in request["instructions"]
    # Dependency files are read at the clone's root, but the environment does not go there (outside the
    # folder this session may write): it is disposable state outside every Git tree, in the scratch the
    # host owns for a full-access run, else the unowned default. A foreign project hears no safe_test.
    assert f"Dependency files in {clone.resolve()} (sha256): none at its top level" in request["instructions"]
    record = Path(ctx.drive_root) / "task_drives" / "child" / SESSION_SCRATCH_DIRNAME / held.snapshot_id / "scratch.json"
    place = (scratch_fact(granted=json.loads(record.read_text(encoding="utf-8"))["path"]) if access == "full"
             else scratch_fact())
    assert record.exists() is (access == "full")
    assert f"disposable process state ({place}), since this session writes only in " \
           f"{start_folder.resolve()}, a folder of the copy" in request["instructions"]
    assert SNAPSHOT_ENVIRONMENT_NOTE not in request["instructions"] and "safe_test" not in request["instructions"]
    assert f"authority/project root {folder} is a read-only" in request["instructions"]
    assert result["execution_root"] == str(start_folder) and result["authority_target_root"] == str(folder)
    assert "evidence-only investigation" in result["note"] and "apply or reject" not in result["note"]
    # The whole worktree, at its real current bytes, is the checking input: a linked worktree of
    # the source at its snapshot commit (a `.git` FILE, the source's own object database).
    assert Path(snapshot["target_root"]) == project.resolve()
    assert (clone / ".git").is_file() and not (clone / ".git" / "objects").exists()
    assert (clone / "conftest.py").read_text() == "ROOT_FIXTURE = 1\n"
    assert (start_folder / "mod.py").read_text() == "VALUE = 2\n"
    assert (start_folder / "draft.txt").read_text() == "untracked input\n"
    assert (held.authority_source, held.target_root, held.execution_root) == (
        INVESTIGATION_AUTHORITY, str(folder), str(start_folder))
    # The child checks in its folder and names evidence the patch filters exclude.
    (start_folder / "mod.py").write_text("VALUE = 3\n")
    (start_folder / ".review-evidence").mkdir()
    (start_folder / ".review-evidence" / "coverage.xml").write_bytes(b"<coverage line-rate='1'/>\n")
    assert custody.emit(ctx.drive_root, custody.SETTLED, {"run_id": held.run_id, "state": "succeeded"})
    held = custody.replay(ctx.drive_root)[held.run_id]
    try:
        block = capture_terminal_patch_for_drive(ctx.drive_root, held)
    finally:
        custody._CUSTODY.clear()
    assert block["status"] == "ready_with_changes", block
    assert "pkg/component/mod.py" in Path(block["patch_artifact"]).read_text()
    assert block["named_evidence"]["status"] == "retained"
    retained = Path(block["named_evidence"]["evidence_dir"]) / "files" / ".review-evidence" / "coverage.xml"
    assert retained.read_bytes() == b"<coverage line-rate='1'/>\n"
    # The capture is the copy's own result; target drift is no longer read or reported.
    assert not {"target_drift_checked", "target_mutated_during_run", "target_drift_unknown"} & set(block)
    assert block["disposition"] == custody.EVIDENCE_RETAINED and not clone.exists()
    assert find_execution_snapshot(held.snapshot_id) is None
    assert "INTEGRATE_DELEGATED_INVESTIGATION_EVIDENCE" in _integrate_delegated_patch(ctx, held.run_id, "apply", "no")
    assert _live_state(project) == before  # HEAD, status and every byte of the live worktree
    assert _git(project, "for-each-ref", "refs/ouroboros") == ""  # the released clone's baseline pin is gone


def _source_local(project, policy):
    """One repository-local input to how the source's Git compares working files;
    the working bytes keep their committed or staged content."""
    if policy == "filemode":
        _git(project, "config", "core.filemode", "false")
        (project / "conftest.py").chmod(0o755)
        return
    if policy == "autocrlf":
        _git(project, "config", "core.autocrlf", "true")
    else:
        (project / ".git" / "info").mkdir(exist_ok=True)
        # The policy names its line end: a bare `text` checks out the host's native one (CRLF on Windows).
        (project / ".git" / "info" / "attributes").write_text("*.py text eol=lf\n*.txt text eol=lf\n",
                                                              encoding="utf-8", newline="\n")
    for path in (project / "conftest.py", *(project / "pkg" / "component").iterdir()):
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))


def _settled_capture(ctx, held):
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    assert custody.emit(ctx.drive_root, custody.SETTLED, {"run_id": held.run_id, "state": "succeeded"})
    try:
        return capture_terminal_patch_for_drive(ctx.drive_root, custody.replay(ctx.drive_root)[held.run_id])
    finally:
        custody._CUSTODY.clear()


@pytest.mark.parametrize("policy", ["autocrlf", "info_attributes", "filemode"])
@pytest.mark.parametrize("access", ["full", "workspace_write"])
@pytest.mark.parametrize("geometry", ["root", "subdirectory"])
def test_source_local_comparison_settings_reach_the_investigation_copy(tmp_path, monkeypatch, geometry, access, policy):
    from ouroboros.subagent_worktrees import find_execution_snapshot

    project, folder = _project(tmp_path)
    _source_local(project, policy)
    before, config = _live_state(project), (project / ".git" / "config").read_bytes()
    index = _git(project, "ls-files", "--stage")
    ctx, engine, start = _consumer(tmp_path, monkeypatch, access, workspace=project if geometry == "root" else folder)
    result = start()
    assert result["status"] == "started" and len(engine.starts) == 1, result
    held = custody.replay(ctx.drive_root)[result["run_id"]]
    clone = Path(find_execution_snapshot(held.snapshot_id)["path"])
    # The copy holds the source's current files as Git checks them out under the source's own
    # (shared) settings, and its Git sees them unchanged. autocrlf and filemode reproduce the
    # working bytes; the `text eol=lf` attributes check normalized files out with LF.
    copied = {rel: (clone / rel).read_bytes() for rel in before[2]}
    if policy == "info_attributes":
        assert {rel: raw.replace(b"\r\n", b"\n") for rel, raw in before[2].items()} == copied
    else:
        assert copied == before[2]
    assert _git(clone, "status", "--porcelain", "--untracked-files=all") == ""
    # One edited line is one ordinary patch line, never a whole-file rewrite.
    mod = clone / "pkg" / "component" / "mod.py"
    mod.write_bytes(mod.read_bytes().replace(b"VALUE = 2", b"VALUE = 3"))
    block = _settled_capture(ctx, held)
    assert block["status"] == "ready_with_changes", block
    patch = Path(block["patch_artifact"]).read_bytes()
    assert patch.count(b"diff --git") == 1 and b"-VALUE = 2\n+VALUE = 3\n" in patch
    assert b"\r" not in patch and b"mode" not in patch
    assert not {"target_drift_checked", "target_mutated_during_run"} & set(block)
    assert block["disposition"] == custody.EVIDENCE_RETAINED and not clone.exists()
    assert _live_state(project) == before and (project / ".git" / "config").read_bytes() == config
    assert _git(project, "ls-files", "--stage") == index


@pytest.mark.parametrize("policy", ["autocrlf", "info_attributes", "filemode"])
def test_an_unchanged_copy_captures_nothing_while_the_source_moves_on(tmp_path, monkeypatch, policy):
    """The copy is a snapshot: a source edit made after it reaches neither the copy nor its capture,
    and the capture reads no target drift (the owner's edit is the owner's, not the run's)."""
    from ouroboros.subagent_worktrees import find_execution_snapshot

    project, folder = _project(tmp_path)
    _source_local(project, policy)
    ctx, engine, start = _consumer(tmp_path, monkeypatch, "full", workspace=folder)
    result = start()
    assert result["status"] == "started", result
    held = custody.replay(ctx.drive_root)[result["run_id"]]
    clone = Path(find_execution_snapshot(held.snapshot_id)["path"])
    copied = (clone / "conftest.py").read_bytes()
    mod = folder / "mod.py"
    mod.write_bytes(mod.read_bytes())  # same bytes, new stat
    (project / "conftest.py").write_bytes(b"ROOT_FIXTURE = 2\r\n")  # the owner's real edit
    assert (clone / "conftest.py").read_bytes() == copied
    block = _settled_capture(ctx, held)
    manifest = json.loads(Path(block["manifest_artifact"]).read_text())
    assert block["status"] == "ready_no_changes" and block["patch_artifact"] is None, block
    assert manifest["patch_size"] == 0 and manifest["tracked_changed"] == []
    assert manifest["result_commit"] and _git(project, "rev-parse", f"{manifest['result_commit']}^{{tree}}") \
        == _git(project, "rev-parse", f"{held.baseline_sha}^{{tree}}")
    assert not {"target_drift_checked", "target_mutated_during_run", "target_drift_unknown"} & set(block)
    assert block["disposition"] == custody.EVIDENCE_RETAINED and not clone.exists()
    assert (project / "conftest.py").read_bytes() == b"ROOT_FIXTURE = 2\r\n"  # the source keeps its edit


def test_git_subdirectory_continuation_of_a_retained_investigation_starts_in_a_fresh_copy_of_the_folder(
        tmp_path, monkeypatch):
    """Retained evidence of any size releases the copy (no 64 MiB bound keeps it); the continuation
    then starts from a fresh copy of the current tree, again in the matching folder."""
    from ouroboros.subagent_worktrees import find_execution_snapshot
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    project, folder = _project(tmp_path)
    ctx, engine, start = _consumer(tmp_path, monkeypatch, "workspace_write", workspace=folder)
    first = start()
    assert first["status"] == "started", first
    held = custody.replay(ctx.drive_root)[first["run_id"]]
    start_folder = Path(held.execution_root)
    (start_folder / ".review-evidence").mkdir()
    (start_folder / ".review-evidence" / "large.log").write_bytes(b"x" * 256)
    assert custody.emit(ctx.drive_root, custody.SETTLED, {"run_id": held.run_id, "state": "succeeded"})
    held = custody.replay(ctx.drive_root)[held.run_id]
    block = capture_terminal_patch_for_drive(ctx.drive_root, held)
    assert block["named_evidence"]["status"] == "retained" and block["disposition"] == custody.EVIDENCE_RETAINED
    assert not start_folder.exists() and find_execution_snapshot(held.snapshot_id) is None
    custody._CUSTODY.clear()
    second = start(continue_from=held.run_id)
    assert second["status"] == "started", second
    assert second["continuation"]["workspace"] == "fresh"
    successor = custody.replay(ctx.drive_root)[second["run_id"]]
    fresh = Path(find_execution_snapshot(successor.snapshot_id)["path"])
    assert successor.snapshot_id != held.snapshot_id and fresh != start_folder.parent.parent
    assert engine.starts[1]["execution"]["workspaceRoot"] == str(fresh / "pkg" / "component") == successor.execution_root
    assert (fresh / "pkg" / "component" / "mod.py").read_text() == "VALUE = 2\n"
    custody._CUSTODY.clear()


@pytest.mark.parametrize("policy", ["", "autocrlf"])
def test_git_subdirectory_readonly_control_reads_the_live_folder(tmp_path, monkeypatch, policy):
    project, folder = _project(tmp_path)
    if policy:
        _source_local(project, policy)
    before = _live_state(project)
    ctx, engine, start = _consumer(tmp_path, monkeypatch, "readonly", workspace=folder)
    result = start()
    assert result["status"] == "started", result
    [request] = engine.starts
    assert (request["access"], request["mode"]) == ("readonly", "ask") and "execution" not in request
    assert request["scope"]["root"] == str(folder) and result["root"] == str(folder)
    held = custody.replay(ctx.drive_root)[result["run_id"]]
    assert not held.snapshot_id and not (tmp_path / "snaps").exists()
    assert _live_state(project) == before
    custody._CUSTODY.clear()


def _folderless(tmp_path):
    data = tmp_path / "data"
    parent, own = data / "task_drives" / "parent", data / "task_drives" / "child"
    parent.mkdir(parents=True)
    own.mkdir(parents=True)
    (parent / "brief.md").write_text("parent brief\n")
    (own / "notes.txt").write_text("own notes\n")
    return own, {"project_id": "prj-folderless"}


def _scratch_files(scratch):
    return {str(p.relative_to(scratch)): p.read_bytes() for p in sorted(scratch.rglob("*"))
            if p.is_file() and "delegated_readonly_inputs" not in p.relative_to(scratch).parts}


@pytest.mark.parametrize("access", ["full", "workspace_write"])
def test_folderless_project_investigation_copies_the_staged_inputs(tmp_path, monkeypatch, access):
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    scratch, metadata = _folderless(tmp_path)
    before = _scratch_files(scratch)
    ctx, engine, start = _consumer(tmp_path, monkeypatch, access, metadata=metadata)
    assert ctx.active_repo_dir() == scratch
    result = start()
    assert result["status"] == "started", result
    [request] = engine.starts
    staged = Path(request["scope"]["root"])
    assert staged.parent == scratch / "delegated_readonly_inputs" and result["root"] == str(staged)
    assert request["scope"] == {"kind": "project", "root": str(staged), "ephemeral": True}
    assert engine.registered == [] and engine.grants == ([str(staged)] if access == "full" else [])
    assert (request["access"], request["mode"]) == (access, "agent")
    assert request["execution"] == {"isolation": "envelope", "delegated": True,
                                    "workspaceKind": "directory", "scopePaths": ["."]}
    manifest = {row["source"]: row for row in json.loads((staged / "inputs.json").read_text())}
    for source in (scratch / "notes.txt", tmp_path / "data" / "task_drives" / "parent" / "brief.md"):
        row = manifest[str(source.resolve())] if str(source.resolve()) in manifest else manifest[str(source)]
        assert (staged / row["local"]).read_bytes() == source.read_bytes()
    for text in ("READONLY INPUTS", "DELEGATED DIRECTORY COPY BINDING", INVESTIGATION_INSTRUCTION):
        assert text in request["instructions"]
    # The copy is its retained evidence: no sentence may deny the folder its PURPOSE line names.
    assert "no evidence folder is retained" not in request["instructions"]
    assert f"outside this copy goes outside every Git working tree ({scratch_fact()})" in request["instructions"]
    assert result["directory_strategy"] == "copy" and result["authority_target_root"] == str(scratch)
    assert "evidence" in result["note"] and "apply or reject" not in result["note"]
    held = custody.replay(ctx.drive_root)[result["run_id"]]
    assert (held.authority_source, held.target_root) == (INVESTIGATION_AUTHORITY, str(scratch))
    assert held.resource_ref["source_root"] == str(staged)
    held.settled = True
    try:
        block = capture_terminal_patch_for_drive(ctx.drive_root, held, gateway=engine)
    finally:
        custody._CUSTODY.clear()
    assert block["disposition"] == custody.EVIDENCE_RETAINED, block
    assert engine.decisions == [{"action": "discard"}]
    [output] = [row for row in block["file_outputs"] if row.get("source_path") == ".review-evidence/notes.txt"]
    assert Path(output["path"]).read_bytes() == engine.body
    assert "INTEGRATE_DELEGATED_INVESTIGATION_EVIDENCE" in _integrate_delegated_patch(ctx, held.run_id, "apply", "no")
    assert _scratch_files(scratch) == before


def test_folderless_project_readonly_control_keeps_the_ask_scratch(tmp_path, monkeypatch):
    scratch, metadata = _folderless(tmp_path)
    ctx, engine, start = _consumer(tmp_path, monkeypatch, "readonly", metadata=metadata)
    result = start()
    assert result["status"] == "started", result
    [request] = engine.starts
    assert (request["access"], request["mode"]) == ("readonly", "ask") and "execution" not in request
    staged = Path(request["scope"]["root"])
    assert staged.parent == scratch / "delegated_readonly_inputs" and (staged / "inputs.json").is_file()
    assert "READONLY INPUTS" in request["instructions"] and "DIRECTORY COPY" not in request["instructions"]
    held = custody.replay(ctx.drive_root)[result["run_id"]]
    assert not held.authority_source and not held.target_root and not held.resource_ref
    custody._CUSTODY.clear()


def test_ordinary_folder_investigation_note_never_offers_apply(tmp_path, monkeypatch):
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / "report.md").write_text("draft\n")
    ctx, engine, start = _consumer(tmp_path, monkeypatch, "full", workspace=folder)
    result = start()
    assert result["status"] == "started", result
    assert engine.starts[0]["scope"]["root"] == str(folder)
    assert "evidence" in result["note"] and "apply or reject" not in result["note"]
    custody._CUSTODY.clear()


def test_a_data_drive_subfolder_names_the_readonly_repair(tmp_path, monkeypatch):
    folder = tmp_path / "data" / "logs"
    folder.mkdir(parents=True)
    ctx, engine, start = _consumer(tmp_path, monkeypatch, "full", workspace=folder)
    refused = start()
    assert refused["reason"] == "directory_execution_unavailable" and refused["definitely_unrun"] is True
    assert "access='readonly'" in refused["detail"] and not engine.starts
    custody._CUSTODY.clear()


def test_a_starting_folder_outside_the_snapshot_is_refused_and_released(tmp_path, monkeypatch):
    from ouroboros.subagent_worktrees import _records

    project, _folder = _project(tmp_path)
    (project / ".gitignore").write_text("build/\n")
    ignored = project / "build" / "out"
    ignored.mkdir(parents=True)
    (ignored / "log.txt").write_text("generated\n")
    ctx, engine, start = _consumer(tmp_path, monkeypatch, "full", workspace=ignored)
    refused = start()
    assert refused["reason"] == "execution_snapshot_failed" and refused["definitely_unrun"] is True
    assert "build/out" in refused["detail"] and "access='readonly'" in refused["detail"]
    assert not engine.starts and _records() == []
    assert _git(project, "for-each-ref", "refs/ouroboros") == ""
    assert not [p for p in (tmp_path / "snaps").rglob("*") if p.is_dir() and p.name != COPIES]  # no copy is left
    assert _git(project, "worktree", "list", "--porcelain").count("worktree ") == 1  # no admin dir left
    custody._CUSTODY.clear()
