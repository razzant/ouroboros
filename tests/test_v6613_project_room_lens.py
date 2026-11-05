"""Project-room conversation tools share one selected physical target.

Native edits need no Git repository or promotion. Workspace tasks and children
keep their own bindings; missing room folders never select the system repo.
"""

from __future__ import annotations



from ouroboros.tools.registry import ToolContext



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


# --- writes: all ordinary tools share the room target ------------------------------


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


# --- affordance map + context fact: stated rule == actual surface ------------------


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
