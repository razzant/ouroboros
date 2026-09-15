"""The reserved shared topics resolve to the global shelf from any room.

`overview` is the orientation every context loads, `improvement-backlog` is the
P7 SSOT, and `patterns` is the Pattern Register, whose only writer (post-task
reflection) and whose readers (context, deep self-review, the headless copy) all
use the canonical drive. Before this, a project room silently minted
`projects/<id>/knowledge/overview.md`: the write succeeded, nothing read it, and
the resident orientation slot stayed empty. Ordinary topics keep following the
room they were written in.
"""

from __future__ import annotations

import pytest

from ouroboros import knowledge as store
from ouroboros.tools import knowledge as tools
from ouroboros.tools.registry import ToolContext


def project_ctx(tmp_path, project_id: str = "demo") -> ToolContext:
    """A project room whose canonical root is the temp drive, as a real task has."""
    return ToolContext(repo_dir=tmp_path, drive_root=tmp_path,
                       budget_drive_root=str(tmp_path), project_id=project_id, task_id="t1")


def test_reserved_set_is_exactly_the_three_shared_topics():
    assert tools.GLOBAL_ONLY_TOPICS == frozenset({
        tools.BACKLOG_TOPIC, store.OVERVIEW_TOPIC, tools.PATTERNS_TOPIC})


def test_pattern_register_topic_reaches_the_drive_the_writer_and_readers_use(tmp_path):
    """The Pattern Register has one home, and it is the one everybody else uses.

    The register's writer (`reflection._update_patterns`) and its readers
    (`context.py`, `deep_self_review`, the headless copy) all address
    `memory/knowledge/patterns.md` on the canonical drive. A project room that
    could mint `projects/<id>/knowledge/patterns.md` through the tool would be
    writing error-class learning into a file none of them ever open.
    """
    ctx = project_ctx(tmp_path)
    canonical = tmp_path / "memory" / "knowledge" / "patterns.md"

    written = tools._knowledge_write(
        ctx, topic=tools.PATTERNS_TOPIC,
        content="# Pattern Register\n\n| Error class | Count |\n|---|---|\n| x | 1 |\n")

    assert "✅" in written
    assert canonical.exists()
    assert not (tmp_path / "projects" / "demo" / "knowledge" / "patterns.md").exists()
    assert "| x | 1 |" in tools._knowledge_read(ctx, tools.PATTERNS_TOPIC)


@pytest.mark.parametrize("topic", sorted(tools.GLOBAL_ONLY_TOPICS))
@pytest.mark.parametrize("scope", ["", "global", "project:demo"])
def test_reserved_topics_resolve_globally_from_a_project_room(tmp_path, topic, scope):
    address = tools._address(project_ctx(tmp_path), topic, scope)

    assert address.scope == "global"
    assert address.project_id == ""
    assert address.path == tmp_path / "memory" / "knowledge" / f"{topic}.md"


def test_ordinary_topic_still_follows_the_room(tmp_path):
    """The narrowing is exactly three names — every other topic keeps both shelves."""
    ctx = project_ctx(tmp_path)

    room = tools._address(ctx, "deploy-recipes")
    assert room.scope == "project:demo"
    assert room.path == tmp_path / "projects" / "demo" / "knowledge" / "deploy-recipes.md"
    assert tools._address(ctx, "deploy-recipes", "global").scope == "global"


def test_overview_written_from_a_project_room_is_read_back_globally(tmp_path):
    ctx = project_ctx(tmp_path)

    written = tools._knowledge_write(
        ctx, topic="overview", content="# What I understand\n\nA short orientation.\n")

    assert "✅" in written
    assert (tmp_path / "memory" / "knowledge" / "overview.md").exists()
    assert not (tmp_path / "projects" / "demo" / "knowledge" / "overview.md").exists()
    assert "A short orientation." in tools._knowledge_read(ctx, "overview")


def test_light_nominations_inherit_the_rule(tmp_path):
    """Light nominates through `_write_knowledge_entries`, which shares `_address`,
    so the consolidator cannot mint a per-project overview either."""
    from ouroboros.consolidator import _write_knowledge_entries

    ctx = project_ctx(tmp_path)
    shelf = tmp_path / "projects" / "demo" / "knowledge"

    shared, room = _write_knowledge_entries(shelf, [
        {"topic": "overview", "content": "# Orientation\n\nShared.\n"},
        {"topic": "room-notes", "content": "# Room\n\nProject detail.\n"},
    ], context=ctx)

    assert (shared["topic"], shared["scope"], shared["ok"]) == ("overview", "global", True)
    assert (room["topic"], room["scope"], room["ok"]) == ("room-notes", "project:demo", True)
    assert (tmp_path / "memory" / "knowledge" / "overview.md").exists()
    assert (shelf / "room-notes.md").exists()


def test_bound_nomination_addresses_carry_the_global_scope(tmp_path):
    """`bind_entries` stamps the address the host will actually write to, so a
    model-supplied project scope for a reserved topic is recorded as global."""
    from ouroboros.consolidator import KnowledgeReadContext

    binder = KnowledgeReadContext(project_ctx(tmp_path))

    bound = binder.bind_entries([
        {"topic": "overview", "content": "x", "scope": "project:demo"},
        {"topic": "room-notes", "content": "y"},
    ])

    assert [(row["topic"], row["scope"]) for row in bound] == [
        ("overview", "global"), ("room-notes", "project:demo")]


def test_missing_project_note_points_at_the_global_shelf(tmp_path):
    ctx = project_ctx(tmp_path)

    missing = tools._knowledge_read(ctx, "deploy-recipes")
    assert "not found in project:demo" in missing
    assert "scope='global'" in missing
    # No false hint when the global shelf is the one that was just searched.
    assert "may exist" not in tools._knowledge_read(ctx, "deploy-recipes", scope="global")
    assert "may exist" not in tools._knowledge_read(ctx, "overview")
    # Still a pure read: a miss creates no shelf, no index, no lock.
    assert not (tmp_path / "projects").exists()
    assert not (tmp_path / "memory").exists()


def test_backlog_merge_semantics_survive_the_shared_resolver(tmp_path):
    """The backlog keeps its own merge path; only its address is now shared."""
    ctx = project_ctx(tmp_path)
    item = "### ibl-1\n- summary: Trim the index preview.\n- category: memory\n"

    merged = tools._knowledge_write(ctx, topic=tools.BACKLOG_TOPIC, content=item)

    assert "merged into the global backlog" in merged
    assert (tmp_path / "memory" / "knowledge" / f"{tools.BACKLOG_TOPIC}.md").exists()
    assert not (tmp_path / "projects" / "demo" / "knowledge").exists()


def test_write_schema_states_the_rules_the_code_actually_enforces(tmp_path):
    """The schema is the SSOT of the per-tool contract, so its sentences are
    assertions about behaviour, not decoration."""
    schemas = {entry.name: entry.schema for entry in tools.get_tools()}
    props = schemas["knowledge_write"]["parameters"]["properties"]

    assert "Reserved topics (improvement-backlog, overview, patterns) always resolve to global." in props["scope"]["description"]
    assert "people and relationships" in props["scope"]["description"]
    assert "no scope prefixes" in props["topic"]["description"]
    assert "stays resident in the index" in schemas["knowledge_write"]["description"]

    # …and the retained-frontmatter claim is true: a body-only overwrite keeps
    # the authored summary that the index renders.
    ctx = project_ctx(tmp_path)
    tools._knowledge_write(ctx, topic="people/ada", scope="global",
                           content="---\ntype: note\nsummary: Prefers short answers.\n---\n\nFirst.\n")
    address = tools._address(ctx, "people/ada", "global")
    revision = store.read_knowledge_note(address).revision
    tools._knowledge_write(ctx, topic="people/ada", scope="global",
                           content="Second, revised.\n", expected_revision=revision)

    note = store.read_knowledge_note(address)
    assert note.summary == "Prefers short answers."
    assert "Second, revised." in note.text
