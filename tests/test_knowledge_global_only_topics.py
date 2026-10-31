"""Three topics resolve to the global shelf from any room.

`overview` is the orientation every context loads. `patterns` and
`improvement-backlog` are ordinary notes kept at their old global address, so an
earlier call from a project room still reaches the one existing note. Before
this, a project room silently minted `projects/<id>/knowledge/overview.md`: the
write succeeded, nothing read it, and the resident orientation slot stayed
empty. Ordinary topics keep following the room they were written in.
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


def test_the_former_register_keeps_its_one_global_home(tmp_path):
    """The former register keeps one home: a call from a project room reaches the
    existing global note instead of minting a second, project-local one."""
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


def test_the_backlog_is_one_authored_global_note_from_a_project_room(tmp_path):
    """The backlog is an ordinary revision-checked note; only its address is reserved."""
    ctx = project_ctx(tmp_path)
    item = "- Trim the index preview.\n"

    written = tools._knowledge_write(ctx, topic=tools.BACKLOG_TOPIC, content=item)

    assert written.startswith("✅")
    path = tmp_path / "memory" / "knowledge" / f"{tools.BACKLOG_TOPIC}.md"
    assert path.read_text(encoding="utf-8").endswith(item)  # the author's text, not a re-rendered item list
    assert not (tmp_path / "projects" / "demo" / "knowledge").exists()
    # A second overwrite without the current revision is refused, never merged.
    stale = tools._knowledge_write(ctx, topic=tools.BACKLOG_TOPIC, content="- Another item.\n")
    assert "revision_required" in stale and path.read_text(encoding="utf-8").endswith(item)


def test_write_schema_states_the_rules_the_code_actually_enforces(tmp_path):
    """The schema is the SSOT of the per-tool contract, so its sentences are
    assertions about behaviour, not decoration."""
    schemas = {entry.name: entry.schema for entry in tools.get_tools()}
    props = schemas["knowledge_write"]["parameters"]["properties"]

    assert "The topics overview, patterns and improvement-backlog always resolve to global." in props["scope"]["description"]
    assert "The overview always stays active." in props["mode"]["description"]
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
