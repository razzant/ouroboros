"""Lessons live where they act: no privileged register in every context.

The Pattern Register and the improvement backlog are ordinary notes. Their bodies
ride in no context (the index carries their summaries), they archive and restore like
any note while keeping their old global address, and only the overview must stay
active. The frozen post-task reflection log renders nowhere; its files stay for
their readers. A write tells the author where the note now sits in its contexts.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import context
from ouroboros import knowledge as store
from ouroboros.headless import _copy_stable_memory
from ouroboros.improvement_backlog import BACKLOG_READER, authored_backlog_text
from ouroboros.memory import Memory
from ouroboros.tools import knowledge as tools
from ouroboros.tools.registry import ToolContext

REGISTER_SENTINEL = "REGISTER_BODY_SENTINEL every class and its counts"


def _env(tmp_path):
    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir(parents=True, exist_ok=True)
    Memory(drive_root=drive, repo_dir=repo).ensure_files()
    return SimpleNamespace(repo_dir=repo, drive_root=drive, repo_path=lambda p: repo / p,
                           drive_path=lambda p: drive / p)


def _ctx(root, **fields):
    return ToolContext(repo_dir=root, drive_root=root, budget_drive_root=str(root), task_id="t1", **fields)


def _write(ctx, topic, body, summary=None, **kwargs):
    meta = f"summary: {summary}\n" if summary else ""
    return tools._knowledge_write(ctx, topic=topic, content=f"---\ntype: note\n{meta}---\n{body}\n", **kwargs)


def _revision(ctx, topic):
    text = tools._knowledge_read(ctx, topic, "global")
    return json.loads(text.split("\n", 1)[0].removeprefix("[Knowledge source] "))["revision"]


def test_the_register_body_rides_in_no_context_but_its_summary_and_reader_do(tmp_path):
    env = _env(tmp_path)
    store.write_knowledge_note(store.resolve_knowledge_address(env.drive_root, "patterns"),
                               f"---\ntype: note\nsummary: Failure classes I have met and what closed them.\n---\n"
                               f"# Pattern Register\n\n{REGISTER_SENTINEL}\n")
    sections = "\n".join(context.build_knowledge_sections(env))
    assert REGISTER_SENTINEL not in sections and "## Known error patterns" not in sections
    # The index keeps its one line: the title and the authored summary.
    assert "**patterns**: [Pattern Register](<patterns.md>)\n  Failure classes I have met and what closed them." in sections
    assert REGISTER_SENTINEL in tools._knowledge_read(_ctx(env.drive_root), "patterns", "global")


def test_only_the_overview_must_stay_active(tmp_path):
    ctx = _ctx(tmp_path)
    for topic in ("patterns", "improvement-backlog"):
        assert "✅" in _write(ctx, topic, "old classes", summary="s")
        archived = tools._knowledge_write(ctx, topic=topic, mode="archive", reason="closed by the change that removed it",
                                          expected_revision=_revision(ctx, topic))
        assert "✅" in archived and "archived" in archived
        restored = tools._knowledge_write(ctx, topic=topic, mode="restore", expected_revision=_revision(ctx, topic))
        assert "✅" in restored
    assert "✅" in _write(ctx, "overview", "what holds now", summary="s")
    refused = tools._knowledge_write(ctx, topic="overview", mode="archive", reason="r",
                                     expected_revision=_revision(ctx, "overview"))
    assert "TOOL_ARG_ERROR" in refused and "overview must stay active" in refused
    assert store.ALWAYS_ACTIVE_TOPICS == frozenset({"overview"})


def test_old_global_addresses_hold_while_new_project_topics_stay_in_their_room(tmp_path):
    ctx = _ctx(tmp_path, project_id="demo")
    assert tools._address(ctx, "patterns").scope == "global"
    assert tools._address(ctx, "improvement-backlog", "project:demo").scope == "global"
    assert tools._address(ctx, "lessons/connectors").scope == "project:demo"


def test_the_index_marks_a_missing_summary_and_nothing_else(tmp_path):
    rows = ({"topic": "with", "title": "With", "summary": "When this applies.", "state": "active"},
            {"topic": "without", "title": "Without", "summary": "", "state": "active"},
            {"topic": "gone", "title": "Gone", "summary": "", "state": "archived",
             "archive": {"at": "2026-10-23", "reason": "closed"}})
    text = store.render_knowledge_index(rows, view="all")
    assert "**without**: [Without](<without.md>) (no summary)" in text
    assert text.count("(no summary)") == 1
    assert "When this applies." in text


def test_a_write_says_where_the_note_now_sits(tmp_path):
    ctx = _ctx(tmp_path)
    summarized = _write(ctx, "lessons/tests", "Read the exit code.", summary="When running tests.")
    assert "summary (19 chars) is shown in the global knowledge index" in summarized
    assert "read on demand" in summarized
    bare = _write(ctx, "lessons/bare", "No summary here.")
    assert "it has no summary" in bare and "(no summary)" in bare
    overview = _write(ctx, "overview", "What holds now.", summary="s")
    # The receipt names the integrating contexts that hold the overview whole and the helpers that read it
    # on demand (BIBLE P1, the subject of continuity).
    assert "loaded whole in every integrating context" in overview and "helpers read on demand" in overview
    archived = tools._knowledge_write(ctx, topic="lessons/bare", mode="archive", reason="moved into a test",
                                      expected_revision=_revision(ctx, "lessons/bare"))
    assert "not in the active knowledge index" in archived
    project = _write(_ctx(tmp_path, project_id="demo"), "lessons/local", "Project lesson.", summary="In demo.")
    assert "project:demo knowledge index, loaded in that project's contexts" in project


def test_the_frozen_reflection_log_renders_nowhere(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "task_reflections.jsonl").write_text(json.dumps({
        "ts": "2026-10-01T00:00:00Z", "task_id": "t0", "reflection": "OLD_AUTOMATIC_LESSON"}) + "\n", encoding="utf-8")
    (tmp_path / "logs" / "supervisor.jsonl").write_text(json.dumps({
        "ts": "2026-10-01T00:00:00Z", "type": "worker_spawn", "task_id": "t0"}) + "\n", encoding="utf-8")
    combined = "\n\n".join(context.build_recent_sections(Memory(drive_root=tmp_path), env=None, task_id="t0"))
    assert "Execution reflections" not in combined and "OLD_AUTOMATIC_LESSON" not in combined
    assert (tmp_path / "logs" / "task_reflections.jsonl").exists()


def test_the_backlog_adapter_names_each_state_without_inventing_one(tmp_path):
    assert authored_backlog_text(tmp_path) == "No improvement-backlog note exists."
    ctx = _ctx(tmp_path)
    _write(ctx, "improvement-backlog", "ITEM_BODY one open idea", summary="Ideas.")
    active = authored_backlog_text(tmp_path)
    assert "ITEM_BODY one open idea" in active and BACKLOG_READER in active and "complete" in active
    tools._knowledge_write(ctx, topic="improvement-backlog", mode="archive", reason="moved into notes",
                           expected_revision=_revision(ctx, "improvement-backlog"))
    archived = authored_backlog_text(tmp_path)
    assert "archived" in archived and "moved into notes" in archived and "ITEM_BODY" not in archived
    path = tmp_path / "memory" / "knowledge" / "improvement-backlog.md"
    path.write_bytes(b"\xff\xfe broken")
    assert "unknown, not empty" in authored_backlog_text(tmp_path)


@pytest.mark.parametrize("project_id", ["", "demo"])
def test_a_project_child_starts_without_a_register_copy(tmp_path, project_id):
    parent, child = tmp_path / "parent", tmp_path / "child"
    (parent / "memory" / "knowledge").mkdir(parents=True)
    (parent / "memory" / "identity.md").write_text("me", encoding="utf-8")
    (parent / "memory" / "knowledge" / "patterns.md").write_text("register", encoding="utf-8")
    _copy_stable_memory(parent, child, project_id=project_id)
    assert (child / "memory" / "identity.md").read_text(encoding="utf-8") == "me"
    # A non-project child inherits the whole knowledge shelf; a project child inherits none of it.
    assert (child / "memory" / "knowledge" / "patterns.md").exists() is (not project_id)


def test_the_prompt_and_the_constitution_name_learning_not_a_register():
    """The words the mind lives by carry the failure-time check and no privileged store."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    prompt = " ".join((repo / "prompts" / "SYSTEM.md").read_text(encoding="utf-8").split())
    bible = " ".join((repo / "BIBLE.md").read_text(encoding="utf-8").split())
    assert "Before closing a failure I check what I already know about its class" in prompt
    assert "a mark that keeps it in view or meets me at the action it concerns" in prompt
    assert "Pattern Register" not in prompt and "improvement backlog or a scratchpad line" not in prompt
    assert "Learning changes what happens next." in bible and "**Durable learning**" in bible
    for gone in ("Pattern Register", "two-strike", "patterns.md", "improvement-backlog.md", "If three behavioral rules"):
        assert gone not in bible, gone


def test_the_residency_fact_fits_inside_the_receipt_bound(tmp_path):
    """A rewrite that removes many headings still reports success: the residency fact is added
    before the receipt's size bound, which then drops the heading list instead of failing."""
    from ouroboros.tools.registry import ToolRegistry

    body = "\n".join(f"## {'Ж' * 29} {i}\ntext {i}\n" for i in range(40))
    ctx = _ctx(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ctx)
    assert "✅" in registry.execute("knowledge_write", {"topic": "big", "content": body})
    for title_len in range(1, 60):
        topic = f"big{title_len}"
        assert "✅" in registry.execute("knowledge_write", {
            "topic": topic, "content": "\n".join(f"## {'Ж' * title_len} {i}\ntext\n" for i in range(40))})
        rewritten = registry.execute("knowledge_write", {"topic": topic, "content": "short body\n",
                                                         "expected_revision": _revision(ctx, topic)})
        assert "✅" in rewritten and "knowledge_residency" in rewritten, (title_len, rewritten[:200])
