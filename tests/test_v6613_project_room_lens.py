"""v6.61.3 — project-room chat lens (affordance-context coherence).

The robot-room incident: a folder-room's DIRECT-CHAT lane resolved ``"."``
against the system repo while the room fact named the project folder — the
agent listed the wrong tree and narrated it as the project. The lens re-points
the chat lane's active_workspace READS and the default shell cwd at the room's
registered working_dir; writes stay with promoted tasks (typed refusal).

These tests pin BOTH sides: the lens where it must activate, and byte-identical
old behavior everywhere else (workspace tasks, subagents, file-less rooms,
headless/bench shapes never carry the lens key).
"""

from __future__ import annotations

import json
import pathlib

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
    assert project_room_lens_dir(ctx5) is None  # missing dir: lens off, never a guess


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


# --- writes: typed refusal pointing at the promote path ----------------------------

def test_room_writes_refused_with_promote_hint(tmp_path):
    from ouroboros.tools.core import _edit_text, _write_file

    ctx, room, repo = _room_ctx(tmp_path)
    out = _write_file(ctx, path="game.js", content="hack")
    assert "ROOM_WRITE_VIA_TASK" in out and "promote_chat_to_task" in out
    assert (room / "game.js").read_text(encoding="utf-8") == "// game\n"
    assert not (repo / "game.js").exists()  # the silent-repo-write trap is closed

    out2 = _edit_text(ctx, path="game.js", old_str="// game", new_str="// hacked")
    assert "ROOM_WRITE_VIA_TASK" in out2
    assert (room / "game.js").read_text(encoding="utf-8") == "// game\n"


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


def test_room_claude_code_edit_default_cwd_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from ouroboros.tools.shell import _claude_code_edit

    ctx, _, _ = _room_ctx(tmp_path)
    out = _claude_code_edit(ctx, prompt="fix everything")
    assert "ROOM_WRITE_VIA_TASK" in out


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
    assert resolved2 == "" and "unusable" in note2  # loud, never a silent repo fallback


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
    assert "LOOKS AT the project folder" in rendered

    update_project(data, "p1", working_dir=str(tmp_path / "vanished"))
    rendered2 = build_runtime_section(env, task)
    assert "working_dir_warning" in rendered2 and "unusable" in rendered2
