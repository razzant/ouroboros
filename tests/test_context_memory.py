"""The open conversation in a built request and the memory sections around it.

Split out of ``tests/test_context.py`` by theme. This module owns where the open
conversation starts (after the legacy frontier), Main's lines for other live rooms
against a Project task's own room, the owner-selected Low that keeps lane 1 verbatim,
the workpad/journal that may not be silently sliced, the world profile, the process
logs filtered by task id, and the installed-skills verdict. The old chat tail after a
consolidation cursor is gone (the memory view's projection: ``tests/test_memory_view_memo.py``).
"""

from __future__ import annotations

import json


def test_the_open_conversation_starts_after_the_legacy_frontier(tmp_path):
    """Rows the old memory retold are its record's, not the open conversation's; A2A never enters."""
    from tests import _memory_inventory_shared as shared
    from tests._memory_view_context import blocks, section, world

    env, memory, _rooms = world(tmp_path)
    shared.append(memory.drive_root / "logs" / "chat.jsonl",
                  shared.msg("2026-09-03T00:10:00+00:00", "agent traffic after", chat_id=-5))
    _a, identity, changing, _cap = blocks(env, memory, {"id": "tmain", "chat_id": 1})
    room = section(changing, "## This room (Main)")
    assert "] next please" in room and "] and more" in room  # after the frontier: verbatim
    # before it: the retold records of this room, whole; the first block's in my story, not repeated on the page
    assert "\n  Main talk." in section(identity, "## My story") and "Main talk." not in room
    assert "Main was quiet." in room
    for retold in ("] hello\n", "] hello back", "alpha question", "agent traffic"):
        assert retold not in changing, retold


def test_main_sees_other_rooms_as_lines_and_a_project_task_its_own_room(tmp_path):
    """One mind: Main names every live room in a line; a task bound to a Project works in that room."""
    from tests._memory_view_context import blocks, section, world

    env, memory, rooms = world(tmp_path)
    alpha = f"Project Alpha [chat_id={rooms['alpha']}]"
    _a, _b, main, _cap = blocks(env, memory, {"id": "tmain", "chat_id": 1})
    live = section(main, "## Live rooms")
    assert alpha in live and f"Project Beta [chat_id={rooms['beta']}]" in live
    assert "alpha again" not in main and "beta again" not in main  # other rooms' words stay lines
    assert "] next please" in section(main, "## This room (Main)")

    _a, identity, bound, _cap = blocks(env, memory, {"id": "bound", "chat_id": 1})  # bound to alpha, written in Main
    room = section(bound, f"## This room ({alpha})")
    assert "] alpha again" in room and "Alpha worked." in room
    assert "\n  Alpha began." in section(identity, "## My story") and "Alpha began." not in room  # first block: story
    assert "next please" not in bound and "beta again" not in bound
    assert "### Main — open" in section(bound, "## Live rooms")


def test_low_mode_keeps_the_open_conversation_verbatim(tmp_path, monkeypatch):
    """An owner-selected Low never takes my replies or people's words: only the window could."""
    from tests import _memory_inventory_shared as shared
    from tests._memory_view_context import blocks, section, world

    env, memory, _rooms = world(tmp_path)
    fresh = [shared.msg(f"2026-09-04T{i // 60:02d}:{i % 60:02d}:00+00:00", f"fresh-{i}", client_message_id=f"f{i}")
             for i in range(305)]
    shared.append(memory.drive_root / "logs" / "chat.jsonl", *fresh)
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")
    _a, _b, changing, cap = blocks(env, memory, {"id": "tmain", "chat_id": 1})
    room = section(changing, "## This room (Main)")
    assert all(f"] fresh-{i}\n" in room + "\n" for i in range(305))
    assert cap["memory_view"]["floor"]["mode"] == "low"
    assert not {"F2", "F6", "F7"} & set(cap["memory_view"]["floor"]["steps"])


def test_project_workpad_and_journal_not_silently_sliced(tmp_path, monkeypatch):
    """BIBLE P1 (no silent truncation): project cognitive artifacts are not
    prefix-sliced into context. The workpad rides in FULL; journal milestones show
    full text (no per-row [:N]) with a visible journal_read pointer for older."""
    import types

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros.context import build_knowledge_sections
    from ouroboros.project_facts import project_journal_path, project_workpad_path
    from ouroboros.utils import append_jsonl

    pid = "builder"
    wp = project_workpad_path(pid)
    wp.parent.mkdir(parents=True, exist_ok=True)
    tail = "WORKPAD_TAIL_MARKER"
    wp.write_text("A" * 20_000 + tail, encoding="utf-8")  # > old 12_000 slice
    append_jsonl(project_journal_path(pid), {
        "ts": "2026-06-14T00:00:00Z", "kind": "checkpoint", "text": "M" * 600,  # > old 200 slice
    })

    env = types.SimpleNamespace(drive_path=lambda rel: tmp_path / rel)
    combined = "\n\n".join(build_knowledge_sections(env, project_id=pid))

    assert tail in combined          # full workpad, not prefix-sliced to 12_000
    assert ("M" * 600) in combined   # full journal milestone, not sliced to 200


def test_append_journal_milestone_bounds_over_limit_with_pointer(tmp_path, monkeypatch):
    """An AUTOMATIC completion milestone honors the journal's durable per-row cap:
    over-limit text is bounded with a VISIBLE pointer (recorded, never silently
    sliced nor dropped) — same _MAX_TEXT_CHARS contract as the journal_write tool,
    so emit_task_results cannot append a raw unbounded row."""
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros.project_facts import project_journal_path
    from ouroboros.tools.project_journal import _MAX_TEXT_CHARS, append_journal_milestone
    from ouroboros.utils import iter_jsonl_objects

    pid = "lh"
    append_journal_milestone(pid, "done", "Z" * (_MAX_TEXT_CHARS + 500), task_id="t1")
    rows = [r for r in iter_jsonl_objects(project_journal_path(pid)) if isinstance(r, dict)]
    assert len(rows) == 1                      # recorded (not dropped/rejected)
    txt = rows[0]["text"]
    assert len(txt) <= _MAX_TEXT_CHARS         # honors the durable per-row contract
    assert "task_results" in txt               # VISIBLE pointer to the full text


def test_world_profile_is_loaded_with_stable_memory(tmp_path):
    from ouroboros.context import build_memory_sections
    from ouroboros.memory import Memory

    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "WORLD.md").write_text("world-profile-data", encoding="utf-8")
    memory = Memory(drive_root=tmp_path)

    sections = build_memory_sections(memory)
    combined = "\n\n".join(sections)

    assert "world-profile-data" in combined


def test_recent_sections_filter_process_logs_by_task_id(tmp_path):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "progress.jsonl").write_text(
        "\n".join([
            json.dumps({"task_id": "task-a", "text": "in-scope"}),
            json.dumps({"task_id": "task-b", "text": "out-of-scope"}),
        ]) + "\n",
        encoding="utf-8",
    )
    (logs_dir / "tools.jsonl").write_text(
        "\n".join([
            json.dumps({"task_id": "task-a", "tool": "shell"}),
            json.dumps({"task_id": "task-b", "tool": "shell"}),
        ]) + "\n",
        encoding="utf-8",
    )

    memory = Memory(drive_root=tmp_path)
    sections = build_recent_sections(memory, env=None, task_id="task-a")
    combined = "\n\n".join(sections)
    assert "in-scope" in combined
    assert "out-of-scope" not in combined


def test_installed_skills_section_includes_warnings_verdict(tmp_path, monkeypatch):
    from ouroboros.context import _build_installed_skills_section

    class FakeEnv:
        drive_root = tmp_path

    monkeypatch.setattr(
        "ouroboros.skill_loader.summarize_skills",
        lambda _root: {
            "skills": [
                {
                    "name": "weather",
                    "type": "script",
                    "enabled": True,
                    "review_status": "warnings",
                    "executable_review": True,
                    "review_stale": False,
                    "description": "Weather helper",
                }
            ]
        },
    )

    section = _build_installed_skills_section(FakeEnv())

    assert "## Installed Skills" in section
    assert "weather" in section
    assert "warnings" in section


def test_installed_skills_section_qualifies_extension_liveness_process(tmp_path, monkeypatch):
    from ouroboros.context import _build_installed_skills_section

    class FakeEnv:
        drive_root = tmp_path

    monkeypatch.setattr(
        "ouroboros.skill_loader.summarize_skills",
        lambda _root: {
            "skills": [{
                "name": "weather_widget",
                "type": "extension",
                "enabled": True,
                "review_status": "pass",
                "executable_review": True,
                "review_stale": False,
                "live_loaded": False,
                "live_reason": "load_error",
                "process": "worker",
            }],
        },
    )

    section = _build_installed_skills_section(FakeEnv())

    assert "Live (worker): no (load_error)" in section
