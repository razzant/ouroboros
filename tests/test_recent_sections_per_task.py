"""Recent-activity sections are each task's OWN newest rows (razzant/ouroboros#131).

Two-sided pins: the interleaved case that the global-tail-then-filter reader
lost, the quiet single-task case that must render exactly as before, the
review-marker window the tools quota must keep, and the coverage line the
header must carry (BIBLE P1: a bounded window is disclosed, never silent).
"""

from __future__ import annotations

import json
import pathlib

from ouroboros.context import build_recent_sections
from ouroboros.memory import Memory


def _write(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _interleaved(root: pathlib.Path) -> None:
    """Task A's rows sit BEFORE a burst of 250 rows from a busy neighbour."""
    tools = [{"ts": f"2026-09-21T20:00:{i % 60:02d}", "task_id": "task-a", "tool": "read_file",
              "args": {"path": f"a-{i}.py"}, "result_preview": "ok"} for i in range(30)]
    tools += [{"ts": "2026-09-21T21:00:00", "task_id": "task-b", "tool": "run_command",
               "args": {"cmd": f"b-{i}"}, "result_preview": "ok"} for i in range(250)]
    progress = [{"ts": "t", "task_id": "task-a", "text": f"a-step-{i}"} for i in range(60)]
    progress += [{"ts": "t", "task_id": "task-b", "text": f"b-step-{i}"} for i in range(250)]
    events = [{"ts": "t", "task_id": "task-a", "type": "llm_round"} for _ in range(40)]
    events += [{"ts": "t", "task_id": "task-b", "type": "tool_error", "error": "boom"} for _ in range(250)]
    _write(root / "logs" / "tools.jsonl", tools)
    _write(root / "logs" / "progress.jsonl", progress)
    _write(root / "logs" / "events.jsonl", events)


def _section(sections, header):
    return next(s for s in sections if s.startswith(header))


def test_task_sees_its_own_newest_rows_behind_a_busy_neighbour(tmp_path):
    _interleaved(tmp_path)
    sections = build_recent_sections(Memory(drive_root=tmp_path), env=None, task_id="task-a")
    tools = _section(sections, "## Recent tools")
    assert "a-29.py" in tools and "a-20.py" in tools and "b-" not in tools
    progress = _section(sections, "## Recent progress")
    assert "a-step-59" in progress and "a-step-10" in progress and "b-step" not in progress
    events = _section(sections, "## Recent events")
    assert "llm_round: 40" in events and "tool_error" not in events
    # The guard's other side: the retired global-tail reader misses every A row.
    stale = [e for e in Memory(drive_root=tmp_path).read_jsonl_tail("tools.jsonl", 200)
             if e.get("task_id") == "task-a"]
    assert stale == []


def test_single_task_log_renders_exactly_as_before(tmp_path):
    rows = [{"ts": "t", "task_id": "task-a", "tool": "shell", "args": {"cmd": f"c{i}"},
             "result_preview": "ok"} for i in range(5)]
    _write(tmp_path / "logs" / "tools.jsonl", rows)
    memory = Memory(drive_root=tmp_path)
    tools = _section(build_recent_sections(memory, env=None, task_id="task-a"), "## Recent tools")
    assert tools.split("\n\n", 1)[1] == memory.summarize_tools(rows)
    assert "task task-a: all 5 matching calls; window: whole live file of logs/tools.jsonl" in tools.splitlines()[0]


def test_no_task_id_keeps_the_global_tail(tmp_path):
    rows = [{"ts": "t", "task_id": f"task-{i % 3}", "text": f"row-{i}"} for i in range(30)]
    _write(tmp_path / "logs" / "progress.jsonl", rows)
    progress = _section(build_recent_sections(Memory(drive_root=tmp_path), env=None), "## Recent progress")
    assert "row-29" in progress and "row-0" in progress
    assert "all tasks: all 30 matching rows" in progress.splitlines()[0]


def test_review_marker_inside_rows_eleven_to_twenty_survives(tmp_path):
    rows = [{"ts": "t", "task_id": "task-a", "tool": "commit_reviewed", "args": {},
             "result_preview": "REVIEW_BLOCKED: tests red"}]
    rows += [{"ts": "t", "task_id": "task-a", "tool": "read_file", "args": {"path": f"f{i}"},
              "result_preview": "ok"} for i in range(15)]
    rows += [{"ts": "t", "task_id": "task-b", "tool": "x", "args": {}, "result_preview": "ok"}
             for _ in range(300)]
    _write(tmp_path / "logs" / "tools.jsonl", rows)
    tools = _section(build_recent_sections(Memory(drive_root=tmp_path), env=None, task_id="task-a"), "## Recent tools")
    assert "REVIEW_FAIL commit_reviewed" in tools


def test_coverage_line_discloses_bounded_archives_and_gaps(tmp_path):
    logs = tmp_path / "logs"
    archive = tmp_path / "archive"
    for i in range(5):
        _write(archive / f"tools_2026090{i}T000000.jsonl",
               [{"ts": "t", "task_id": "task-a", "tool": "t", "args": {}, "result_preview": "ok"}])
    _write(logs / "tools.jsonl", [{"ts": "t", "task_id": "task-a", "tool": "live", "args": {}, "result_preview": "ok"}])
    rows, coverage = Memory(drive_root=tmp_path).read_task_recent("tools.jsonl", "task-a", 20)
    assert len(rows) == 4 and coverage["archives"] == 3 and coverage["archives_available"] == 5
    assert coverage["archives_bounded"] and not coverage["quota_met"]
    header = _section(build_recent_sections(Memory(drive_root=tmp_path), env=None, task_id="task-a"),
                      "## Recent tools").splitlines()[0]
    assert "3 of 5 newest archives" in header and "older archives not opened" in header
    assert "of logs/tools.jsonl" in header


def test_unreadable_archive_directory_is_a_disclosed_gap(tmp_path, monkeypatch):
    """Portable stand-in for an EACCES archive directory (chmod is not a Windows fact)."""
    import os as _os

    _write(tmp_path / "logs" / "tools.jsonl", [{"ts": "t", "task_id": "task-a", "tool": "live", "args": {}, "result_preview": "ok"}])
    (tmp_path / "archive").mkdir()
    real_scandir = _os.scandir

    def denied(path, *args, **kwargs):
        if str(path).endswith("archive"):
            raise PermissionError(13, "denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(_os, "scandir", denied)
    rows, coverage = Memory(drive_root=tmp_path).read_task_recent("tools.jsonl", "task-a", 20)
    assert [r["tool"] for r in rows] == ["live"] and coverage["gaps"] == ["unreadable_source"]


def test_coverage_line_never_says_all_under_a_bounded_window(tmp_path):
    from ouroboros.jsonl_tail import TAIL_WINDOW_START_BYTES, coverage_line

    padded = [{"ts": "t", "task_id": "task-b", "text": "x" * 4000} for _ in range(400)]  # > 512 KB
    padded += [{"ts": "t", "task_id": "task-a", "text": f"a-{i}"} for i in range(50)]
    _write(tmp_path / "logs" / "progress.jsonl", padded)
    rows, coverage = Memory(drive_root=tmp_path).read_task_recent("progress.jsonl", "task-a", 50)
    assert len(rows) == 50 and coverage["live_window"] == TAIL_WINDOW_START_BYTES
    line = coverage_line(coverage)
    assert "all " not in line and "newest 50 matching rows in the window" in line
    assert "live tail 512 KB of" in line and "of logs/progress.jsonl" in line
    assert coverage_line({"task_id": "t", "shown": 0, "matched": 0, "gaps": ["read_failed"]}).endswith(
        "window: unread; gaps: read_failed")


def test_reader_never_parses_the_whole_live_file_when_the_tail_suffices(tmp_path, monkeypatch):
    from ouroboros import jsonl_tail

    rows = [{"ts": "t", "task_id": "task-a", "text": "x" * 2000} for _ in range(1500)]  # ~3 MB
    _write(tmp_path / "logs" / "progress.jsonl", rows)
    windows = []
    original = jsonl_tail.iter_jsonl_objects

    def spy(path, *args, **kwargs):
        windows.append(kwargs.get("tail_bytes"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(jsonl_tail, "iter_jsonl_objects", spy)
    shown, coverage = Memory(drive_root=tmp_path).read_task_recent("progress.jsonl", "task-a", 50)
    assert len(shown) == 50 and windows == [jsonl_tail.TAIL_WINDOW_START_BYTES]
    assert coverage["live_window"] == jsonl_tail.TAIL_WINDOW_START_BYTES < coverage["live_size"]


def test_malformed_only_log_still_discloses_its_gap(tmp_path):
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "logs" / "tools.jsonl").write_text("{not json}\n", encoding="utf-8")
    sections = build_recent_sections(Memory(drive_root=tmp_path), env=None, task_id="task-a")
    tools = _section(sections, "## Recent tools")
    assert "no matching calls" in tools and "gaps: malformed_jsonl" in tools
    # A log with neither rows nor gaps stays silent, as before.
    (tmp_path / "logs" / "tools.jsonl").write_text("", encoding="utf-8")
    assert not [s for s in build_recent_sections(Memory(drive_root=tmp_path), env=None, task_id="task-a")
                if s.startswith("## Recent tools")]


def test_supervisor_section_carries_its_coverage_line(tmp_path):
    _write(tmp_path / "logs" / "supervisor.jsonl", [{"ts": "2026-09-22T00:00:00Z", "type": "boot", "branch": "ouroboros", "sha": "abcdef123456"}])
    section = _section(build_recent_sections(Memory(drive_root=tmp_path), env=None), "## Supervisor")
    assert section.splitlines()[0].startswith("## Supervisor (all tasks: all 1 matching rows; window: whole live file of logs/supervisor.jsonl")
    assert "boot: 2026-09-22T00:00:00Z branch=ouroboros sha=abcdef123456" in section


def test_bounded_window_with_no_matching_rows_is_still_disclosed(tmp_path):
    """The task's only row sits in the fourth-oldest archive: the window is empty
    AND incomplete, and the section must say so (scope finding, 2026-09-22)."""
    archive = tmp_path / "archive"
    _write(archive / "tools_20260901T000000.jsonl",
           [{"ts": "t", "task_id": "task-a", "tool": "old", "args": {}, "result_preview": "ok"}])
    for i in range(2, 5):
        _write(archive / f"tools_2026090{i}T000000.jsonl",
               [{"ts": "t", "task_id": "task-b", "tool": "b", "args": {}, "result_preview": "ok"}])
    _write(tmp_path / "logs" / "tools.jsonl", [{"ts": "t", "task_id": "task-b", "tool": "live", "args": {}, "result_preview": "ok"}])
    tools = _section(build_recent_sections(Memory(drive_root=tmp_path), env=None, task_id="task-a"), "## Recent tools")
    header = tools.splitlines()[0]
    assert "no matching calls" in header and "older archives not opened" in header and "3 of 4 newest archives" in header
    assert tools.strip() == header  # nothing rendered below the disclosure


def test_tools_header_says_how_many_rows_are_rendered(tmp_path):
    rows = [{"ts": "t", "task_id": "task-a", "tool": "shell", "args": {"cmd": f"c{i}"}, "result_preview": "ok"}
            for i in range(30)]
    _write(tmp_path / "logs" / "tools.jsonl", rows)
    tools = _section(build_recent_sections(Memory(drive_root=tmp_path), env=None, task_id="task-a"), "## Recent tools")
    assert "newest 20 of 30 matching calls in the window (10 rendered, 20 scanned for review markers)" in tools.splitlines()[0]
    assert tools.count("shell cmd=") == 10


def test_unreadable_live_file_is_disclosed_as_unread(tmp_path, monkeypatch):
    """A live log that cannot be read is never called "whole live file" (BIBLE P1)."""
    _write(tmp_path / "logs" / "tools.jsonl",
           [{"ts": "t", "task_id": "task-a", "tool": "shell", "args": {}, "result_preview": "ok"}])
    real_stat = pathlib.Path.stat

    def denied(self, *args, **kwargs):
        if self.name == "tools.jsonl":
            raise PermissionError(13, "simulated EACCES", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "stat", denied)
    header = _section(build_recent_sections(Memory(drive_root=tmp_path), env=None, task_id="task-a"),
                      "## Recent tools").splitlines()[0]
    assert "window: unread of logs/tools.jsonl" in header and "unreadable_source" in header
    assert "whole live file" not in header and "no matching calls" in header
    monkeypatch.setattr(pathlib.Path, "stat", real_stat)
    # A log that was never written is an empty window, not a gap: no section at all.
    (tmp_path / "logs" / "tools.jsonl").unlink()
    sections = build_recent_sections(Memory(drive_root=tmp_path), env=None, task_id="task-a")
    assert not any(s.startswith("## Recent tools") for s in sections)


def test_no_task_headers_do_not_overclaim_what_was_scanned(tmp_path):
    """Without a task id the global 200-row tail is kept; progress renders 50 of it and
    tools scans only its newest 20 for review markers, and the headers say exactly that."""
    _write(tmp_path / "logs" / "progress.jsonl", [{"ts": "t", "task_id": "t", "text": f"p{i}"} for i in range(260)])
    _write(tmp_path / "logs" / "tools.jsonl",
           [{"ts": "t", "task_id": "t", "tool": "shell", "args": {"cmd": f"c{i}"}, "result_preview": "ok"} for i in range(260)])
    sections = build_recent_sections(Memory(drive_root=tmp_path), env=None)
    progress = _section(sections, "## Recent progress").splitlines()[0]
    assert "(newest 50 rendered of 200 loaded)" in progress and "review markers" not in progress
    tools = _section(sections, "## Recent tools").splitlines()[0]
    assert "(10 rendered, 20 scanned for review markers)" in tools and "200 scanned" not in tools


def _repo_and_drive(tmp_path):
    repo_dir = tmp_path / "repo"
    drive_root = tmp_path / "drive"
    (repo_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (repo_dir / "docs").mkdir(parents=True, exist_ok=True)
    (drive_root / "memory" / "knowledge").mkdir(parents=True, exist_ok=True)
    (drive_root / "logs").mkdir(parents=True, exist_ok=True)
    (drive_root / "state").mkdir(parents=True, exist_ok=True)
    (repo_dir / "prompts" / "SYSTEM.md").write_text("System prompt", encoding="utf-8")
    (repo_dir / "BIBLE.md").write_text("Bible", encoding="utf-8")
    (repo_dir / "VERSION").write_text("1.2.3", encoding="utf-8")
    (repo_dir / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
    (repo_dir / "README.md").write_text("README", encoding="utf-8")
    (repo_dir / "docs" / "ARCHITECTURE.md").write_text("# Ouroboros v1.2.3", encoding="utf-8")
    (repo_dir / "docs" / "DEVELOPMENT.md").write_text(
        "### File Size Budgets\n| Path | Budget chars |\n|------|--------------|\n| memory/identity.md | 1000 |\n",
        encoding="utf-8",
    )
    (drive_root / "state" / "state.json").write_text('{"spent_usd": 0, "budget_drift_alert": false}', encoding="utf-8")
    (drive_root / "memory" / "identity.md").write_text("x" * 950, encoding="utf-8")
    (drive_root / "memory" / "scratchpad.md").write_text("scratchpad", encoding="utf-8")
    for name in ("chat", "supervisor", "task_reflections"):
        (drive_root / "logs" / f"{name}.jsonl").write_text("", encoding="utf-8")

    class FakeEnv:
        def drive_path(self, p):
            return drive_root / p

        def repo_path(self, p):
            return repo_dir / p

        @property
        def repo_dir(self):
            return repo_dir

        @property
        def drive_root(self):
            return drive_root

    return repo_dir, drive_root, FakeEnv()


def test_child_reads_its_own_drive_beside_working_sources(tmp_path):
    """A subagent gets its process memory from its execution drive (owner decision 2026-09-22)."""
    from ouroboros.context import build_llm_messages

    repo_dir, canonical, env = _repo_and_drive(tmp_path)
    child_drive = tmp_path / "child"
    (child_drive / "logs").mkdir(parents=True)
    # Canonical logs: the parent's traffic plus the child's mirrored tool rows and its progress.
    _write(canonical / "logs" / "tools.jsonl",
           [{"ts": "t", "task_id": "parent", "tool": "parent_tool", "args": {}, "result_preview": "ok"}] * 5
           + [{"ts": "t", "task_id": "child", "tool": "mirrored", "args": {}, "result_preview": "ok"}])
    _write(canonical / "logs" / "events.jsonl",
           [{"ts": "t", "task_id": "child", "type": "delegate_supervision_wait_renewed"}] * 3)
    _write(canonical / "logs" / "progress.jsonl",
           [{"ts": "t", "task_id": "parent", "text": "parent-step"}, {"ts": "t", "task_id": "child", "text": "child-step"}])
    # The child's own drive: exactly its worker rows.
    _write(child_drive / "logs" / "tools.jsonl",
           [{"ts": "t", "task_id": "child", "tool": "own_tool", "args": {"path": "x.py"}, "result_preview": "ok"}])
    _write(child_drive / "logs" / "events.jsonl", [{"ts": "t", "task_id": "child", "type": "llm_round"}] * 4)

    messages, _cap = build_llm_messages(
        env=env, memory=Memory(drive_root=child_drive, repo_dir=repo_dir),
        task={"id": "child", "type": "task", "text": "work", "delegation_role": "subagent",
              "parent_task_id": "parent", "root_task_id": "parent", "budget_drive_root": str(canonical)},
    )
    dynamic = messages[0]["content"][2]["text"]
    assert "## Working sources" in dynamic and "Your own recent process" in dynamic
    tools = dynamic[dynamic.index("## Recent tools"):].split("\n## ", 1)[0]
    assert "own_tool" in tools and "parent_tool" not in tools and "mirrored" not in tools
    assert "of task drive logs/tools.jsonl" in tools.splitlines()[0]
    events = dynamic[dynamic.index("## Recent events"):].split("\n## ", 1)[0]
    assert "llm_round: 4" in events and "delegate_supervision" not in events
    assert "host-side rows such as waits stay in the canonical log" in events.splitlines()[0]
    progress = dynamic[dynamic.index("## Recent progress"):].split("\n## ", 1)[0]
    assert "child-step" in progress and "parent-step" not in progress
    assert "of canonical logs/progress.jsonl" in progress.splitlines()[0]
    # The Working sources block precedes the child's own windows.
    assert dynamic.index("## Working sources") < dynamic.index("## Recent progress")


def test_child_without_its_own_drive_reads_the_canonical_filtered_windows(tmp_path):
    from ouroboros.context import build_llm_messages

    repo_dir, canonical, env = _repo_and_drive(tmp_path)
    _write(canonical / "logs" / "tools.jsonl",
           [{"ts": "t", "task_id": "parent", "tool": "parent_tool", "args": {}, "result_preview": "ok"}] * 5
           + [{"ts": "t", "task_id": "child", "tool": "shared_drive_tool", "args": {}, "result_preview": "ok"}])
    for name in ("events", "progress"):
        (canonical / "logs" / f"{name}.jsonl").write_text("", encoding="utf-8")
    messages, _cap = build_llm_messages(
        env=env, memory=Memory(drive_root=canonical, repo_dir=repo_dir),
        task={"id": "child", "type": "task", "text": "work", "delegation_role": "subagent",
              "parent_task_id": "parent", "root_task_id": "parent"},
    )
    dynamic = messages[0]["content"][2]["text"]
    tools = dynamic[dynamic.index("## Recent tools"):].split("\n## ", 1)[0]
    assert "shared_drive_tool" in tools and "parent_tool" not in tools
    assert "of logs/tools.jsonl" in tools.splitlines()[0] and "task drive" not in tools.splitlines()[0]
