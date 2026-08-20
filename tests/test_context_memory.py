"""Recent chat, consolidation offsets and the memory sections around them.

Split verbatim out of ``tests/test_context.py`` by theme (merged there from the former
``test_context_memory_overhaul.py``). This module owns the offset a consolidation
leaves, the full-awareness main thread against a project thread's own view, the
workpad/journal that may not be silently sliced, the low-mode dialogue tail, the stale
offset a rotation invalidates, the world profile, the retired dialogue summaries, the
process logs filtered by task id, and the installed-skills verdict.
"""

from __future__ import annotations

import json





# ===========================================================================
# Memory / consolidation offset behavior (merged from former
# test_context_memory_overhaul.py).  Inspect-only `limit=50` / `limit=1000`
# source-string pins were dropped — behavioral coverage below already
# exercises the offset path.  test_no_identity_truncation_in_consolidator_
# prompts was also dropped (inspect-only); identity-truncation is covered
# behaviorally by consolidator tests.
# ===========================================================================


def test_recent_chat_starts_after_consolidated_offset(tmp_path):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    memory_dir = tmp_path / "memory"
    logs_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {"ts": f"2026-03-19T16:{i:02d}:00Z", "direction": "in", "username": "User", "text": f"msg-{i}"}
        for i in range(5)
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    memory = Memory(drive_root=tmp_path)
    (memory_dir / "dialogue_meta.json").write_text(
        json.dumps({
            "last_consolidated_offset": 3,
            "chat_log_signature": memory.jsonl_generation_signature("chat.jsonl"),
        }),
        encoding="utf-8",
    )

    sections = build_recent_sections(memory, env=None)
    combined = "\n\n".join(sections)

    assert "msg-0" not in combined
    assert "msg-1" not in combined
    assert "msg-2" not in combined
    assert "msg-3" in combined
    assert "msg-4" in combined


def test_recent_chat_main_includes_all_threads_full_awareness(tmp_path):
    """Full project awareness (v6.32.0): the one identity's main/global context
    sees its WHOLE conversation — main + project threads alike (BIBLE P1, one
    awareness across direct chat, project rooms, and consciousness). Project chat
    is part of the one mind's memory, NOT partitioned out; only A2A virtual
    transport is excluded (covered elsewhere)."""
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    project = create_project(tmp_path, "racer")
    project_chat = int(project["chat_id"])
    transport_chat = 555000111  # large NON-project id (e.g. a Telegram mirror)

    entries = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": "main-keep"},
        {"chat_id": project_chat, "direction": "in", "username": "User", "text": "project-visible"},
        {"chat_id": transport_chat, "direction": "in", "username": "User", "text": "transport-keep"},
        {"direction": "in", "username": "User", "text": "legacy-keep"},  # no chat_id -> main
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    combined = "\n\n".join(build_recent_sections(Memory(drive_root=tmp_path), env=None))

    assert "main-keep" in combined
    assert "legacy-keep" in combined
    assert "transport-keep" in combined
    assert "project-visible" in combined  # full awareness: the one mind sees project chat


def test_recent_chat_for_project_thread_shows_only_its_own_thread(tmp_path):
    """A project TASK gets a FOCUSED working view of its own thread (full
    awareness, v6.32.0): its "## Recent chat" is its own project thread, not the
    штаб's main chat nor a sibling project's chat, so cross-project noise does not
    bloat its working context. This is focus, not memory isolation — the one mind
    still sees everything via the main/background path. Pins that thread_chat_id
    selects the project's own raw tail rather than the main consolidation stream."""
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    proj_a = create_project(tmp_path, "racer")
    proj_b = create_project(tmp_path, "research")
    chat_a = int(proj_a["chat_id"])
    chat_b = int(proj_b["chat_id"])

    entries = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": "main-stab-chat"},
        {"chat_id": chat_a, "direction": "in", "username": "User", "text": "project-a-own-thread"},
        {"chat_id": chat_b, "direction": "in", "username": "User", "text": "project-b-sibling"},
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    combined = "\n\n".join(build_recent_sections(
        Memory(drive_root=tmp_path), env=None, thread_chat_id=chat_a))

    assert "project-a-own-thread" in combined   # its own thread is visible
    assert "project-b-sibling" not in combined  # sibling project not in focused view
    assert "main-stab-chat" not in combined     # main chat not in focused project view


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


def test_low_mode_preserves_full_unconsolidated_dialogue_suffix(tmp_path, monkeypatch):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    memory_dir = tmp_path / "memory"
    logs_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    fresh_count = 305
    entries = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": f"consolidated-{i}"}
        for i in range(3)
    ] + [
        {"chat_id": 1, "direction": "in", "username": "User", "text": f"fresh-{i}"}
        for i in range(fresh_count)
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    memory = Memory(drive_root=tmp_path)
    (memory_dir / "dialogue_meta.json").write_text(
        json.dumps({
            "last_consolidated_offset": 3,
            "chat_log_signature": memory.jsonl_generation_signature("chat.jsonl"),
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")

    combined = "\n\n".join(build_recent_sections(memory, env=None))

    assert "consolidated-0" not in combined
    assert "fresh-0" in combined
    assert f"fresh-{fresh_count - 1}" in combined


def test_low_mode_without_consolidation_keeps_max_raw_dialogue_tail(tmp_path, monkeypatch):
    from ouroboros.context import build_recent_sections
    from ouroboros.context_budget import MAX_RECENT_CHAT_TAIL
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fresh_count = 305
    entries = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": f"fresh-{i}"}
        for i in range(fresh_count)
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")

    combined = "\n\n".join(build_recent_sections(Memory(drive_root=tmp_path), env=None))

    assert fresh_count < MAX_RECENT_CHAT_TAIL
    assert "fresh-0" in combined
    assert f"fresh-{fresh_count - 1}" in combined


def test_recent_chat_offset_uses_filtered_dialogue_entries(tmp_path):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    memory_dir = tmp_path / "memory"
    logs_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": "consolidated-0"},
        {"chat_id": -1, "direction": "in", "username": "Agent", "text": "a2a-noise"},
        {"chat_id": 1, "direction": "in", "username": "User", "text": "consolidated-1"},
        {"chat_id": 1, "direction": "in", "username": "User", "text": "fresh"},
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    memory = Memory(drive_root=tmp_path)
    (memory_dir / "dialogue_meta.json").write_text(
        json.dumps({
            "last_consolidated_offset": 2,
            "chat_log_signature": memory.jsonl_generation_signature("chat.jsonl"),
        }),
        encoding="utf-8",
    )

    combined = "\n\n".join(build_recent_sections(memory, env=None))

    assert "consolidated-0" not in combined
    assert "consolidated-1" not in combined
    assert "a2a-noise" not in combined
    assert "fresh" in combined


def test_recent_chat_ignores_stale_consolidation_offset_after_rotation(tmp_path):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    memory_dir = tmp_path / "memory"
    logs_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    initial = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": f"early-{i}"}
        for i in range(3)
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in initial) + "\n",
        encoding="utf-8",
    )
    memory = Memory(drive_root=tmp_path)
    stale_signature = memory.jsonl_generation_signature("chat.jsonl")
    (memory_dir / "dialogue_meta.json").write_text(
        json.dumps({
            "last_consolidated_offset": 3,
            "chat_log_signature": stale_signature,
        }),
        encoding="utf-8",
    )

    rotated = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": f"post-rotate-{i}"}
        for i in range(2)
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in rotated) + "\n",
        encoding="utf-8",
    )

    combined = "\n\n".join(build_recent_sections(memory, env=None))

    # Rotation invalidates the stale offset; rotated entries appear.
    assert "post-rotate-0" in combined
    assert "post-rotate-1" in combined


def test_recent_chat_keeps_offset_when_same_log_gets_appended(tmp_path):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    memory_dir = tmp_path / "memory"
    logs_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    initial = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": f"old-{i}"}
        for i in range(3)
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in initial) + "\n",
        encoding="utf-8",
    )
    memory = Memory(drive_root=tmp_path)
    (memory_dir / "dialogue_meta.json").write_text(
        json.dumps({
            "last_consolidated_offset": 3,
            "chat_log_signature": memory.jsonl_generation_signature("chat.jsonl"),
        }),
        encoding="utf-8",
    )

    with open(logs_dir / "chat.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"chat_id": 1, "direction": "in", "username": "User", "text": "new"}) + "\n")

    combined = "\n\n".join(build_recent_sections(memory, env=None))

    assert "old-0" not in combined
    assert "new" in combined


def test_world_profile_is_loaded_with_stable_memory(tmp_path):
    from ouroboros.context import build_memory_sections
    from ouroboros.memory import Memory

    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "WORLD.md").write_text("world-profile-data", encoding="utf-8")
    memory = Memory(drive_root=tmp_path)

    sections = build_memory_sections(memory)
    combined = "\n\n".join(sections)

    assert "world-profile-data" in combined


def test_retired_dialogue_summary_remains_visible_when_blocks_exist(tmp_path):
    from ouroboros.context import build_memory_sections
    from ouroboros.memory import Memory

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "dialogue_summary.md").write_text("legacy dialogue", encoding="utf-8")
    (memory_dir / "dialogue_blocks.json").write_text(
        json.dumps([{"content": "new dialogue block"}]),
        encoding="utf-8",
    )
    memory = Memory(drive_root=tmp_path)

    combined = "\n\n".join(build_memory_sections(memory, partition="volatile"))

    assert "## Dialogue History" in combined
    assert "new dialogue block" in combined
    assert "## Legacy Dialogue Summary (retired flat format, read-only fallback)" in combined
    assert "legacy dialogue" in combined


def test_retired_dialogue_summary_fallback_preserves_continuity_without_blocks(tmp_path):
    from ouroboros.context import build_memory_sections
    from ouroboros.memory import Memory

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "dialogue_summary.md").write_text("legacy dialogue only", encoding="utf-8")
    memory = Memory(drive_root=tmp_path)

    combined = "\n\n".join(build_memory_sections(memory, partition="volatile"))

    assert "## Legacy Dialogue Summary (retired flat format, read-only fallback)" in combined
    assert "legacy dialogue only" in combined


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
