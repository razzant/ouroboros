"""Physical Markdown addresses remain exact across syntax and note metadata."""

import hashlib

import pytest

from ouroboros.markdown_source import MarkdownSourceError, parse_markdown_source


def test_note_metadata_and_source_bytes_survive_parsing():
    raw = (
        "---\r\ntype: unexpected-kind\r\nsummary: |\r\n  First thought.\r\n"
        "  Second thought.\r\ncustom:\r\n  future: [yes, 3]\r\n---\r\n"
        "# Знакомство\r\n\r\nA note [related](../other%20note.md).\r\n"
    ).encode()
    source = parse_markdown_source(raw, "knowledge/people/one.md")
    assert source.raw == raw
    assert source.text.encode() == raw
    assert source.sha256 == hashlib.sha256(raw).hexdigest()
    assert source.frontmatter["type"] == "unexpected-kind"
    assert source.frontmatter["summary"] == "First thought.\nSecond thought.\n"
    assert source.frontmatter["custom"] == {"future": [True, 3]}
    assert source.text_at(source.body_span).startswith("# Знакомство\r\n")
    assert source.links[0].destination == "../other%20note.md"
    assert source.links[0].span.start_line == 11
    assert source.text_at(source.links[0].span) == "[related](../other%20note.md)"


def test_real_markdown_structure_excludes_code_and_resolves_references():
    raw = b"""# Title

A [direct](a.md) and [Reference][r].

```markdown
## Fake heading
[fake](fake.md)
```

    [indented](also-fake.md)

## Actual section

[Reference][] and [r].

[r]: <folder/b.md>
[Reference]: c.md
"""
    source = parse_markdown_source(raw, "note.md")
    assert [h.title for h in source.headings] == ["Title", "Actual section"]
    assert [link.destination for link in source.links] == ["a.md", "folder/b.md", "c.md", "folder/b.md"]
    for link in source.links:
        assert source.text_at(link.span).startswith("[")
    assert source.headings[0].section.end_byte == len(raw)


def test_physical_lf_lines_do_not_split_unicode_line_separators():
    raw = "# Heading\n\nOne\u2028two\u2029three.\nTail.".encode()
    source = parse_markdown_source(raw, "note.md")
    span = source.lines(3, 1)
    assert source.text_at(span) == "One\u2028two\u2029three.\n"
    assert (span.start_line, span.end_line) == (3, 3)
    assert source.text_at(source.lines(4)) == "Tail."
    with pytest.raises(ValueError):
        source.lines(5)


def test_line_bounds_are_the_lfs_before_a_byte_and_a_large_note_parses_promptly():
    """Every range's lines are the LFs before its first and last byte, whatever the line endings, front matter or
    code points; the LF index is built once per source. The Atlas's 1.5 MB note of 15,000 sections took ~30 s while
    each of its 45,004 ranges recounted the source from byte 0 (issue #2408)."""
    import time

    sources = [b"", b"\n", b"a", b"a\nb", b"a\r\nb\r\n", b"a\rb\r", "# Ж 🌱\r\n\r\n[л](п.md) é\n- x\n  - y".encode(),
               b"---\r\ntitle: t\r\n---\r\n# H\r\n\r\ntext [l](x.md)\r\n```\r\ncode\r\n```", b"---\na: 1\n---",
               b"Title\n=====\n\n> [q](q.md)\n\n[r]: r.md\n[r]\n"]
    for raw in sources:
        source = parse_markdown_source(raw, "n.md")
        spans = [source.body_span, *filter(None, [source.frontmatter_span]), *source.paragraphs, *source.lists,
                 *source.list_items, *source.code_blocks, *(link.span for link in source.links),
                 *(span for heading in source.headings for span in (heading.span, heading.section))]
        for span in spans:
            assert (span.start_line, span.end_line) == (raw.count(b"\n", 0, span.start_byte) + 1,
                                                         raw.count(b"\n", 0, max(span.start_byte, span.end_byte - 1)) + 1)
        physical = raw.decode().split("\n")
        for number in range(1, len(physical) + (0 if raw.endswith(b"\n") else 1)):
            assert source.text_at(source.lines(number, 1)) == physical[number - 1] + ("\n" if number < len(physical) else "")
    raw = ("---\ntitle: Большая\n---\n# Большая\n\n"
           + "## Раздел\n\nТекст большой заметки. Synthetic.\n\n" * 15_000).encode()
    started = time.perf_counter()
    source = parse_markdown_source(raw, "large.md")
    assert time.perf_counter() - started < 5  # ~0.4 s; 20-36 s when each range counted from byte 0
    last = source.headings[-1]
    assert len(source.headings) == 15_001 and (last.span.start_line, last.section.end_line) == (
        raw.count(b"\n", 0, last.span.start_byte) + 1, raw.count(b"\n"))


def test_legacy_markdown_and_setext_heading_have_no_invented_metadata():
    raw = b"Title\n=====\n\nLegacy text.\n"
    source = parse_markdown_source(raw, "legacy.md")
    assert source.frontmatter is None
    assert source.body_span.start_byte == 0
    assert source.headings[0].level == 1
    assert source.headings[0].title == "Title"
    assert source.text_at(source.lines()) == raw.decode()


def test_invalid_source_is_not_repaired_into_plausible_content():
    with pytest.raises(MarkdownSourceError, match="bad.md.*UnicodeDecodeError"):
        parse_markdown_source(b"bad\xff", "bad.md")
    with pytest.raises(ValueError, match="mapping"):
        parse_markdown_source(b"---\n- not-a-map\n---\ntext", "bad.md")


def test_invalid_yaml_is_one_value_error_contract_with_its_original_cause():
    import yaml

    with pytest.raises(MarkdownSourceError, match="broken.md") as caught:
        parse_markdown_source(b"---\nx: [unterminated\n---\n# Body\n", "broken.md")
    assert isinstance(caught.value, ValueError)
    assert isinstance(caught.value.__cause__, yaml.YAMLError)


def test_missing_native_grammar_is_a_source_error_without_module_import_failure(monkeypatch):
    import sys
    from types import SimpleNamespace

    def unavailable(name):
        raise LookupError("native grammar missing")

    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", SimpleNamespace(get_parser=unavailable))
    with pytest.raises(MarkdownSourceError, match="body.md: Markdown parser unavailable.*LookupError"):
        parse_markdown_source(b"# Body\n", "body.md")


def test_absent_native_package_preserves_context_import_and_exact_note_read(tmp_path):
    import pathlib
    import subprocess
    import sys

    script = r'''
import pathlib, sys
sys.modules["tree_sitter_language_pack"] = None
import ouroboros.context
from tests.test_book_context_capture import _capture
from ouroboros.knowledge import resolve_knowledge_address, read_knowledge_note
env, core, task = _capture(pathlib.Path(sys.argv[1]))
assert len(core.reference_book_errors) == 2
assert all("Markdown parser unavailable" in gap for gap in core.reference_book_errors)
assert core.bible_md == "The complete constitution."
assert not core.reference_books
address = resolve_knowledge_address(env.drive_root, "legacy", "global")
address.path.parent.mkdir(parents=True, exist_ok=True)
raw = b"# Legacy\n\nExact complete experience.\n"
address.path.write_bytes(raw)
note = read_knowledge_note(address)
assert note.raw == raw
assert "MarkdownSourceError" in note.parse_error
assert "Markdown parser unavailable" in note.parse_error
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=pathlib.Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_malformed_book_yaml_uses_existing_context_gap_without_losing_identity(tmp_path):
    from tests.test_book_context_capture import _capture, _plan, _text
    from ouroboros import context
    from ouroboros.memory import Memory

    env, _old, task = _capture(tmp_path)
    (env.repo_dir / "docs/architecture/runtime.md").write_bytes(b"---\nx: [broken\n---\n# Runtime\n")
    memory = Memory(drive_root=env.drive_root, repo_dir=env.repo_dir)
    core = context._capture_context_core(env, memory, task, None, None)
    assert core.bible_md == "The complete constitution."
    assert any("docs/architecture/runtime.md" in gap for gap in core.reference_book_errors)
    for mode in ("max", "low", "nano"):
        text = _text(_plan(env, core, task, mode), mode)
        assert "Reference book source unavailable" in text
        assert "My complete identity." in text
        assert "My whole earlier biography." in text
