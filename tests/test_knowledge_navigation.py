"""Current navigation is a cheap projection of files, never a stale index receipt."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import context, knowledge as store, markdown_source
from ouroboros.tools.knowledge import _knowledge_list
from ouroboros.tools.registry import ToolContext


@pytest.mark.parametrize("view", ["active", "archived", "all"])
def test_navigation_matches_full_inventory_with_real_markdown_titles_and_metadata(tmp_path, view):
    address = store.resolve_knowledge_address(tmp_path, "topic")
    address.shelf.mkdir(parents=True)
    sources = {
        "explicit": b"---\ntitle: Authored title\nsummary: Authored summary\n---\n# Different heading\n",
        "fences": b"```md\n# Not a title\n```\n\nReal *heading*\n====\nBody [link](explicit.md)\n",
        "nested": "> ## Заголовок\r\n\r\nBody.\r\n".encode(),
        "archived": b"---\narchive: {at: yesterday, reason: old}\nsummary: Retained\n---\n# History\n",
        "broken-archive": b"---\narchive: null\nsummary: Still active\n---\n# Current\n",
        "broken-yaml": b"---\narchive: [unfinished\n---\n# Current\n",
        "overview": b"---\narchive: {at: yesterday, reason: old}\n---\nOrientation\n",
        "no-heading": b"Plain legacy text.\n",
        "empty-heading": b"#\n",
        "empty-title": b"---\ntitle: ' '\nsummary: [not, text]\n---\n# Heading\n",
        "recursive": b"---\ncustom: &loop [*loop]\n---\n# Recursive\n",
        "bad-utf8": b"---\ntitle: Title\n---\nbad\xff",
    }
    for topic, raw in sources.items():
        (address.shelf / f"{topic}.md").write_bytes(raw)
    expected = store.render_knowledge_index(store.inventory_knowledge(address), view=view)
    assert store.knowledge_index_view(address, view) == expected


@pytest.mark.parametrize("scope", ["global", "project:demo"])
@pytest.mark.parametrize("index_exists", [False, True])
def test_real_consumers_see_external_edits_even_when_size_and_mtime_match(tmp_path, monkeypatch, scope, index_exists):
    import os

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    address = store.resolve_knowledge_address(tmp_path, "topic", scope)
    original = ("---\ntitle: Before\nsummary: Before summary\n"
                "archive: {at: yesterday, reason: old}\n---\n# Body\nFacts.\n")
    assert store.write_knowledge_note(address, "---\ntitle: Before\nsummary: Before summary\n---\n# Body\nFacts.\n").ok
    # Simulate an external producer, outside the write/index transaction.
    address.path.write_text(original)
    if not index_exists:
        (address.shelf / store.INDEX_FILE).unlink()
    env = SimpleNamespace(drive_path=lambda name: tmp_path / name)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="navigation")
    consumers = [lambda: _knowledge_list(ctx, scope=scope),
                 lambda: "\n".join(context.build_knowledge_sections(env, project_id=address.project_id))]
    for consume in consumers:
        assert "Before summary" not in consume()
        assert "Archived notes: 1" in consume()
    stat = address.path.stat()
    # Same-length invalid archive value must become visible; no stat cache may hide it.
    changed = original.replace("Before", "After!").replace("reason: old", "reason: 123")
    assert len(changed.encode()) == len(original.encode())
    address.path.write_text(changed)
    os.utime(address.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    for consume in consumers:
        text = consume()
        assert "After! summary" in text and "source metadata unavailable" in text
    address.path.unlink()
    for consume in consumers:
        assert "After! summary" not in consume()


def test_context_reads_overview_once_and_navigation_skips_full_body_structure(tmp_path, monkeypatch):
    address = store.resolve_knowledge_address(tmp_path, "overview")
    address.shelf.mkdir(parents=True)
    address.path.write_text("---\nsummary: Orientation\n---\nShared understanding.\n")
    (address.shelf / store.INDEX_FILE).write_text("Earlier legacy prose.\n")
    (address.shelf / "details.md").write_text("---\ntitle: Details\nsummary: Meaning\n---\n# Body\n[link](overview.md)\n")
    reads, parsers = [], []
    read_bytes, parser = Path.read_bytes, markdown_source._markdown_parser

    def read(path):
        reads.append(path)
        return read_bytes(path)

    def parse(grammar, path):
        native = parser(grammar, path)

        class ObservedParser:
            def parse(self, raw):
                parsers.append((grammar, path))
                return native.parse(raw)

        return ObservedParser()

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(markdown_source, "_markdown_parser", parse)
    env = SimpleNamespace(drive_path=lambda name: tmp_path / name)
    text = "\n".join(context.build_knowledge_sections(env, include_pattern_body=False))
    assert "Shared understanding." in text and "Meaning" in text
    assert "Earlier legacy prose." not in text
    assert reads.count(address.path) == 1
    assert parsers == [("markdown", str(address.path)), ("markdown_inline", str(address.path))]


@pytest.mark.parametrize("unavailable", ["markdown", "markdown_inline"])
def test_navigation_keeps_native_parser_gaps_visible_for_titled_archived_notes(tmp_path, monkeypatch, unavailable):
    address = store.resolve_knowledge_address(tmp_path, "topic")
    address.shelf.mkdir(parents=True)
    address.path.write_text("---\ntitle: History\narchive: {at: yesterday, reason: old}\n---\n# Body\n")
    parser = markdown_source._markdown_parser

    def missing(grammar, path):
        if grammar == unavailable:
            raise markdown_source.MarkdownSourceError("Native parser unavailable")
        return parser(grammar, path)

    monkeypatch.setattr(markdown_source, "_markdown_parser", missing)
    note = store.read_knowledge_note(address)
    assert note.state == "active" and note.parse_error
    text = store.knowledge_index_view(address)
    assert "**topic**" in text and "source metadata unavailable" in text
