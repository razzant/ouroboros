"""Project-room conversation tools share one selected physical target.

Native edits need no Git repository or promotion. Workspace tasks and children
keep their own bindings; missing room folders never select the system repo.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from ouroboros.tools.registry import ToolContext

pytestmark = pytest.mark.serial


def _room_ctx(tmp_path, *, direct=True, with_room=True, workspace=False):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "BIBLE.md").write_text("repo marker\n", encoding="utf-8")
    drive = tmp_path / "drive"
    drive.mkdir(exist_ok=True)
    room = tmp_path / "robot"
    room.mkdir(exist_ok=True)
    (room / "game.js").write_text("// game\n", encoding="utf-8")
    (room / "index.html").write_text("<html>robot</html>\n", encoding="utf-8")
    meta = {"_project_room_dir": str(room)} if with_room else {}
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=drive,
        system_repo_dir=repo,
        workspace_root=(tmp_path / "ws") if workspace else None,
        workspace_mode="external" if workspace else "",
        task_metadata=meta,
        task_id="t-room",
        is_direct_chat=direct,
    )
    return ctx, room, repo


# --- keying: the lens activates ONLY on the direct-chat folder-room shape ----------

def test_lens_key_requires_all_legs(tmp_path):
    from ouroboros.tool_access import project_room_lens_dir

    ctx, room, _ = _room_ctx(tmp_path)
    assert project_room_lens_dir(ctx) == room.resolve()

    ctx2, _, _ = _room_ctx(tmp_path, direct=False)
    assert project_room_lens_dir(ctx2) is None  # pooled/promoted tasks: never

    ctx3, _, _ = _room_ctx(tmp_path, with_room=False)
    assert project_room_lens_dir(ctx3) is None  # main chat / file-less room

    (tmp_path / "ws").mkdir(exist_ok=True)
    ctx4, _, _ = _room_ctx(tmp_path, workspace=True)
    assert project_room_lens_dir(ctx4) is None  # a task with its OWN workspace

    ctx5, room5, _ = _room_ctx(tmp_path)
    ctx5.task_metadata["_project_room_dir"] = str(tmp_path / "gone-folder")
    assert project_room_lens_dir(ctx5) == (tmp_path / "gone-folder").resolve()


# --- reads resolve to the ROOM folder; self-repo stays reachable explicitly --------

def test_room_reads_resolve_to_room_folder(tmp_path):
    from ouroboros.tools.core import _code_search, _list_files, _read_file

    ctx, room, repo = _room_ctx(tmp_path)

    listing = json.loads(_list_files(ctx, path="."))
    assert "game.js" in listing and "index.html" in listing
    assert "BIBLE.md" not in listing  # the robot incident shape

    body = _read_file(ctx, "game.js")
    assert "// game" in body and "project room" in body

    found = _code_search(ctx, "robot", path=".")
    assert "index.html" in found

    # The system repo remains one EXPLICIT root away (deliberate escape hatch).
    repo_listing = json.loads(_list_files(ctx, path=".", root="system_repo"))
    assert "BIBLE.md" in repo_listing


def test_room_read_confined_to_room(tmp_path):
    from ouroboros.tools.core import _read_file

    ctx, room, repo = _room_ctx(tmp_path)
    escaped = _read_file(ctx, "../repo/BIBLE.md")
    # Traversal out of the room folder must not silently read elsewhere.
    assert "repo marker" not in escaped


def test_fileless_room_and_workspace_task_unchanged(tmp_path):
    from ouroboros.tools.core import _list_files

    ctx, _, repo = _room_ctx(tmp_path, with_room=False)
    listing = json.loads(_list_files(ctx, path="."))
    assert "BIBLE.md" in listing  # old behavior byte-identical

    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    (ws / "app.py").write_text("x\n", encoding="utf-8")
    ctx2, _, _ = _room_ctx(tmp_path, direct=False, workspace=True)
    listing2 = json.loads(_list_files(ctx2, path="."))
    assert "app.py" in listing2  # workspace wiring untouched


# --- writes: all ordinary tools share the room target ------------------------------

@pytest.mark.parametrize("tool", ["edit_text", "edit_batch", "apply_patch"])
def test_room_skill_named_path_does_not_redirect_to_installed_payload(tmp_path, monkeypatch, tool):
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, room, repo = _room_ctx(tmp_path)
    relative = "skills/external/demo/notes.txt"
    room_file, installed_file = room / relative, ctx.drive_root / relative
    for path in (room_file, installed_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("before\n", encoding="utf-8")
    (installed_file.parent / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Fixture\ntype: instruction\n---\nFixture\n",
        encoding="utf-8",
    )
    registry = ToolRegistry(repo, ctx.drive_root)
    registry.set_context(ctx)
    edit = {"path": relative, "old_str": "before", "new_str": "after"}
    arguments = {
        "edit_text": edit,
        "edit_batch": {"edits": [edit]},
        "apply_patch": {"patch": f"*** Begin Patch\n*** Update File: {relative}\n@@\n-before\n+after\n*** End Patch"},
    }[tool]
    result = registry.execute_result(tool, arguments)
    assert result.status == "ok", result.text
    assert room_file.read_text() == "after\n"
    assert installed_file.read_text() == "before\n"
    explicit = registry.execute_result("edit_text", {
        "root": "skill_payload", "bucket": "external", "skill_name": "demo",
        "path": "notes.txt", "old_str": "before", "new_str": "explicit",
    })
    assert explicit.status == "ok", explicit.text
    assert installed_file.read_text() == "explicit\n"
    assert room_file.read_text() == "after\n"


@pytest.mark.parametrize("runtime_mode", ["light", "advanced", "pro"])
@pytest.mark.parametrize("filename", ["game.js", "BIBLE.md"])
def test_room_file_operations_share_an_ordinary_non_git_folder(tmp_path, monkeypatch, runtime_mode, filename):
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", runtime_mode)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, room, repo = _room_ctx(tmp_path)
    system_before = (repo / filename).read_text() if (repo / filename).exists() else None
    registry = ToolRegistry(repo, ctx.drive_root)
    registry.set_context(ctx)
    operations = [
        ("write_file", {"path": str(room / filename), "content": "// written\n"}, "written"),
        ("edit_text", {"path": filename, "old_str": "written", "new_str": "edited"}, "edited"),
        ("edit_batch", {"edits": [{"path": str(room / filename), "old_str": "edited", "new_str": "batched"}]}, "batched"),
        ("apply_patch", {"patch": f"*** Begin Patch\n*** Update File: {filename}\n@@\n-// batched\n+// patched\n*** End Patch"}, "patched"),
    ]
    for name, args, expected in operations:
        assert registry.get_schema_by_name(name) is not None
        result = registry.execute_result(name, args)
        assert result.status == "ok", (name, result.text)
        assert "commit_reviewed" not in result.text and "headless runner" not in result.text
        assert "CORE_PATCH_NOTICE" not in result.text
        assert expected in (room / filename).read_text(encoding="utf-8")
        assert expected in registry.execute("read_file", {"path": filename})
        assert ((repo / filename).read_text() if (repo / filename).exists() else None) == system_before
        assert not (room / ".git").exists()
    shell = registry.execute_result("run_command", {
        "cmd": [sys.executable, "-c", f"from pathlib import Path; Path({filename!r}).write_text('// shell\\n')"],
    })
    assert shell.status == "ok", shell.text
    assert (room / filename).read_text(encoding="utf-8") == "// shell\n"
    assert ((repo / filename).read_text() if (repo / filename).exists() else None) == system_before
    assert (repo / "BIBLE.md").read_text(encoding="utf-8") == "repo marker\n"


def test_missing_room_never_reads_or_edits_the_system_copy(tmp_path):
    from ouroboros.tools.registry import ToolRegistry, active_repo_dir_for
    from ouroboros.tool_access import resource_root_path, resolve_shell_cwd

    ctx, _room, repo = _room_ctx(tmp_path)
    missing = tmp_path / "gone-folder"
    ctx.task_metadata["_project_room_dir"] = str(missing)
    (repo / "game.js").write_text("// system copy\n", encoding="utf-8")
    registry = ToolRegistry(repo, ctx.drive_root)
    registry.set_context(ctx)
    assert active_repo_dir_for(ctx) == resource_root_path(ctx, "active_workspace") == missing
    assert resolve_shell_cwd(ctx)[0] == missing
    assert "system copy" not in registry.execute("read_file", {"path": "game.js"})
    result = registry.execute("edit_text", {"path": "game.js", "old_str": "system copy", "new_str": "changed"})
    assert "file not found" in result.lower() or "no such file" in result.lower(), result
    assert (repo / "game.js").read_text(encoding="utf-8") == "// system copy\n"
    assert "system copy" in registry.execute("read_file", {"path": "game.js", "root": "system_repo"})


def test_unresolved_room_binding_is_a_tool_error_with_explicit_system_reads_available(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    ctx, _room, repo = _room_ctx(tmp_path, with_room=False)
    ctx.task_metadata["_project_room_note"] = "project registry cannot be read"
    registry = ToolRegistry(repo, ctx.drive_root)
    registry.set_context(ctx)
    for name, args in [
        ("read_file", {"path": "BIBLE.md"}),
        ("write_file", {"path": "notes.txt", "content": "should not land"}),
        ("run_command", {"cmd": ["pwd"]}),
    ]:
        result = registry.execute_result(name, args)
        assert result.status != "ok" and "project registry cannot be read" in result.text
    assert not (repo / "notes.txt").exists()
    assert "repo marker" in registry.execute("read_file", {"path": "BIBLE.md", "root": "system_repo"})


@pytest.mark.parametrize("cwd_kind", ["system_repo", "task_drive", "absolute_system"])
def test_unresolved_room_keeps_explicit_authorized_process_roots(tmp_path, monkeypatch, cwd_kind):
    from ouroboros.tool_access import resource_root_path
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, _room, repo = _room_ctx(tmp_path, with_room=False)
    ctx.task_metadata["_project_room_note"] = "project registry cannot be read"
    registry = ToolRegistry(repo, ctx.drive_root)
    registry.set_context(ctx)
    cwd = str(repo) if cwd_kind == "absolute_system" else cwd_kind
    expected = repo if cwd_kind == "absolute_system" else resource_root_path(ctx, cwd_kind)
    result = registry.execute_result("run_command", {
        "cwd": cwd, "cmd": [sys.executable, "-c", "from pathlib import Path; print(Path.cwd())"],
    })
    assert result.status == "ok", result.text
    assert str(expected.resolve()) in result.text


def test_room_delegation_snapshot_capture_and_vcs_share_the_selected_target(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody
    from ouroboros.subagent_worktrees import find_execution_snapshot
    from ouroboros.tools.delegate import _capture_terminal_patch, _derive_authority, _mutation_authority, _provision_snapshot
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_delegated_run_isolation import _isolated_entry, _nanny_ctx, _seed_target

    target = _seed_target(tmp_path)
    ctx = _nanny_ctx(tmp_path, target, monkeypatch)
    ctx.workspace_root, ctx.workspace_mode, ctx.is_direct_chat = None, "", True
    ctx.task_metadata["_project_room_dir"] = str(target)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    registry = ToolRegistry(ctx.repo_dir, ctx.drive_root)
    registry.set_context(ctx)
    shape = _derive_authority(ctx)
    assert shape.access == "workspace_write" and shape.mode == "agent"
    authority, error = _mutation_authority(ctx, shape)
    assert not error and authority["target_root"] == str(target)
    handle, error = _provision_snapshot(ctx, ctx.drive_root, authority["target_root"], "room-snapshot")
    assert not error, error
    execution = pathlib.Path(handle.path)
    assert execution != target and (execution / "tracked.txt").read_text() == "one\ntwo\n"
    try:
        (execution / "from_delegate.txt").write_text("delegated change\n", encoding="utf-8")
        entry = _isolated_entry(ctx, target, handle)
        capture = _capture_terminal_patch(ctx, entry)
        assert capture["authority_target_root"] == str(target)
        assert not (target / "from_delegate.txt").exists()
        result = registry.execute_result("integrate_delegated_patch", {"run_id": "run-1", "decision": "apply"})
        assert result.status == "ok", result.text
        assert (target / "from_delegate.txt").read_text() == "delegated change\n"
        assert not (ctx.repo_dir / "from_delegate.txt").exists()
        status = registry.execute("vcs_status", {})
        assert "from_delegate.txt" in status and f"repo={target}" in status
        assert find_execution_snapshot("room-snapshot") is None
        assert not execution.exists()
    finally:
        custody._CUSTODY.clear()


def test_non_git_room_delegation_reports_its_actual_snapshot_limitation(tmp_path, monkeypatch):
    from ouroboros.tools.delegate import _derive_authority, _mutation_authority, _provision_snapshot

    ctx, room, repo = _room_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snapshots"))
    authority, error = _mutation_authority(ctx, _derive_authority(ctx))
    assert not error and authority["target_root"] == str(room)
    handle, error = _provision_snapshot(ctx, ctx.drive_root, authority["target_root"], "non-git-room")
    refused = json.loads(error.text)
    assert handle is None and refused["reason"] == "execution_snapshot_failed"
    assert refused["target_root"] == str(room) and "not a git working tree" in refused["detail"]
    assert not (room / ".git").exists() and not (repo / ".git").exists()


# --- shell: the DEFAULT cwd is the room folder; explicit cwd still free ------------

def test_room_shell_default_cwd_is_room(tmp_path):
    from ouroboros.tool_access import resolve_shell_cwd

    ctx, room, repo = _room_ctx(tmp_path)
    work_dir, label, allowed = resolve_shell_cwd(ctx, "")
    assert work_dir == room.resolve()
    assert label == "active_workspace"
    # The system repo remains an allowed root for explicit cwds.
    assert any(pathlib.Path(root).resolve() == repo.resolve() for _label, root in allowed)
    explicit, _, _ = resolve_shell_cwd(ctx, str(repo))
    assert explicit == repo.resolve()

    ctx2, _, _ = _room_ctx(tmp_path, with_room=False)
    default2, _, _ = resolve_shell_cwd(ctx2, "")
    assert default2 == repo.resolve()  # non-room chats unchanged


def test_room_first_shell_result_carries_cwd_note(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    from ouroboros.tools.shell import _run_shell

    ctx, room, _ = _room_ctx(tmp_path)
    out = _run_shell(ctx, ["ls"])
    assert "project-room cwd" in out and str(room.resolve()) in out
    assert "game.js" in out
    out2 = _run_shell(ctx, ["ls"])
    assert "project-room cwd" not in out2  # one-shot per task


# --- affordance map + context fact: stated rule == actual surface ------------------

def test_affordance_map_names_room_dir(tmp_path):
    from ouroboros.tool_access import filesystem_affordance_map

    ctx, room, _ = _room_ctx(tmp_path)
    fs = filesystem_affordance_map(ctx)
    assert fs["project_room_dir"] == str(room.resolve())
    assert str(room.resolve()) in fs["default_shell_cwd"]

    ctx2, _, _ = _room_ctx(tmp_path, with_room=False)
    fs2 = filesystem_affordance_map(ctx2)
    assert "project_room_dir" not in fs2


def test_room_chat_lens_dir_resolver(tmp_path):
    from ouroboros.projects_registry import create_project, update_project
    from ouroboros.workspace_admission import room_chat_lens_dir

    data = tmp_path / "data"
    data.mkdir()
    folder = tmp_path / "proj"
    folder.mkdir()
    create_project(data, "p1", name="P1", origin="owner_ui")

    assert room_chat_lens_dir(data, "p1") == ("", "")  # file-less: no lens, no note
    assert room_chat_lens_dir(data, "") == ("", "")
    assert room_chat_lens_dir(data, "ghost") == ("", "")

    update_project(data, "p1", working_dir=str(folder))
    resolved, note = room_chat_lens_dir(data, "p1")
    assert resolved == str(folder.resolve()) and note == ""

    update_project(data, "p1", working_dir=str(tmp_path / "vanished"))
    resolved2, note2 = room_chat_lens_dir(data, "p1")
    assert resolved2 == str(tmp_path / "vanished") and "unusable" in note2


def test_context_fact_matches_lens_state(tmp_path, monkeypatch):
    """The coherence invariant: the room fact's stated rule must match the actual
    tool surface — lens active ⇒ the rule says reads/shell resolve to the folder;
    lens unavailable (broken dir) ⇒ a loud warning rides the fact."""
    from types import SimpleNamespace

    import ouroboros.config as config
    from ouroboros.context import build_runtime_section
    from ouroboros.projects_registry import create_project, update_project

    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    folder = tmp_path / "proj"
    folder.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    create_project(data, "p1", name="P1", origin="owner_ui")
    update_project(data, "p1", working_dir=str(folder))

    env = SimpleNamespace(
        repo_dir=repo,
        drive_root=data,
        drive_path=lambda rel: data / rel,
    )
    task = {"id": "t1", "project_id": "p1", "_is_direct_chat": True}
    rendered = build_runtime_section(env, task)
    assert "active_workspace is working_dir" in rendered

    update_project(data, "p1", working_dir=str(tmp_path / "vanished"))
    rendered2 = build_runtime_section(env, task)
    assert "working_dir_warning" in rendered2 and "unusable" in rendered2
